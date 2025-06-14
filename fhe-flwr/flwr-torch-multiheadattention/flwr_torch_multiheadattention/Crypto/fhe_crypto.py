from openfhe import *

import uuid
import os
import math

import torch
import numpy as np

from logging import INFO
from typing import List, Tuple
import pickle

MULT_DEPTH = 1
SCALE_MOD_SIZE = 50
BATCH_SIZE = 8
BLOCK_SIZE = 4096

class FheCryptoAPI:
    @staticmethod
    def create_crypto_context_and_keys(
        mult_depth=MULT_DEPTH,
        scale_mod_sz=SCALE_MOD_SIZE,
        batch_sz=BATCH_SIZE
    ):
        parameters = CCParamsCKKSRNS()
        parameters.SetMultiplicativeDepth(mult_depth)
        parameters.SetScalingModSize(scale_mod_sz)
        parameters.SetBatchSize(batch_sz)
        parameters.SetRingDim(1 << 16)
        
        cc = GenCryptoContext(parameters)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.KEYSWITCH)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)

        cc_bytes = FheCryptoAPI.serialize_to_bytes(cc)
        pubkey_bytes = FheCryptoAPI.serialize_to_bytes(keys.publicKey)
        seckey_bytes = FheCryptoAPI.serialize_to_bytes(keys.secretKey)

        return cc_bytes, pubkey_bytes, seckey_bytes
    

    @staticmethod
    def __deserialize_bin_file(bytedata: bytes, deserialize_fn):
        filename = f"{uuid.uuid4()}"

        with open(filename, "wb") as f:
            f.write(bytedata)

        obj, res = deserialize_fn(filename, BINARY)

        if not res:
            raise Exception("Unable to deserialize object")
        
        os.remove(filename)
        return obj
    

    @staticmethod
    def serialize_to_bytes(obj):
        filename = f"{uuid.uuid4()}"

        SerializeToFile(filename, obj, BINARY)

        with open(filename, "rb") as f:
            bytedata = f.read()

        os.remove(filename)
        return bytedata


    @staticmethod
    def deserialize_crypto_context(ccbytes: bytes):
        return FheCryptoAPI.__deserialize_bin_file(
            ccbytes,
            DeserializeCryptoContext
        )
    

    @staticmethod
    def deserialize_public_key(keybytes: bytes):
        return FheCryptoAPI.__deserialize_bin_file(
            keybytes,
            DeserializePublicKey
        )
    

    @staticmethod
    def deserialize_private_key(keybytes: bytes):
        return FheCryptoAPI.__deserialize_bin_file(
            keybytes,
            DeserializePrivateKey
        )
    

    @staticmethod
    def deserialize_ciphertext(ciphertext_bytes: bytes):
        return FheCryptoAPI.__deserialize_bin_file(
            ciphertext_bytes,
            DeserializeCiphertext
        )
    

    @staticmethod
    def encrypt_numpy_array(
        cc_bytes: bytes,
        pubkey_bytes: bytes,
        arr: np.array,
        block_sz=BLOCK_SIZE
    ):
        cc = FheCryptoAPI.deserialize_crypto_context(cc_bytes)
        pubkey = FheCryptoAPI.deserialize_public_key(pubkey_bytes)

        enc_arr = []

        arr = arr.flatten()
        for i in range(0, len(arr), block_sz):
            plaintext = cc.MakeCKKSPackedPlaintext(arr[i:i+block_sz], slots=block_sz)
            ciphertext = cc.Encrypt(pubkey, plaintext)
            enc_arr.append(FheCryptoAPI.serialize_to_bytes(ciphertext))

        return enc_arr
    

    @staticmethod
    def decrypt_torch_tensor(
        cc_bytes: bytes,
        seckey_bytes: bytes,
        ciphertext_blocks,
        dtype,
        shape
    ):
        cc = FheCryptoAPI.deserialize_crypto_context(cc_bytes)
        seckey = FheCryptoAPI.deserialize_private_key(seckey_bytes)
        # ciphertext = [FheCryptoAPI.deserialize_ciphertext(ciphertext_blocks)]
        # plaintext = cc.Decrypt(seckey, ciphertext).GetCKKSPackedValue()
        # real_arr = [x.real for x in plaintext]

        decrypted_tensors = []
        for block in ciphertext_blocks:
            cmplx_block = cc.Decrypt(seckey, FheCryptoAPI.deserialize_ciphertext(block)).GetCKKSPackedValue()
            real_tensor = [x.real for x in cmplx_block]
            decrypted_tensors.append(torch.tensor(real_tensor, dtype=dtype))

        flat_sz = math.prod(shape)

        return torch.cat(decrypted_tensors, dim=0)[:flat_sz].reshape(shape).cpu()
    

    @staticmethod
    def aggregate_encrypted(
        encrypted_results: List[Tuple[List[bytes], int]], 
        cc_bytes: bytes) -> List[bytes]:
        """
        Compute encrypted weighted average using CKKS homomorphic encryption.
        
        Args:
            encrypted_results: List of (encrypted_parameters, num_examples) tuples
            cc_bytes: Serialized crypto context
            
        Returns:
            List of encrypted aggregated parameters
        """
        if not encrypted_results:
            return []
        
        cc = FheCryptoAPI.deserialize_crypto_context(cc_bytes)
        
        # Calculate total examples
        num_examples_total = sum(num_examples for (_, num_examples) in encrypted_results)
        
        # Get the first client's encrypted parameters as starting point
        first_encrypted_params, first_num_examples = encrypted_results[0]
        
        # Deserialize first client's ciphertexts
        aggregated_ciphertexts = []
        for encrypted_block_bytes in first_encrypted_params:
            encrypted_blocks = pickle.loads(encrypted_block_bytes)
            # Scale by weight (num_examples / total_examples)
            weight = first_num_examples / num_examples_total
            
            # For each block in the parameter
            weighted_blocks = []
            for block in encrypted_blocks:
                ciphertext = FheCryptoAPI.deserialize_ciphertext(block)
                # Multiply by weight (homomorphic scalar multiplication)
                weighted_ciphertext = cc.EvalMult(ciphertext, weight)
                weighted_blocks.append(FheCryptoAPI.serialize_to_bytes(weighted_ciphertext))
            
            aggregated_ciphertexts.append(weighted_blocks)
        
        # Add remaining clients' weighted parameters
        for encrypted_params, num_examples in encrypted_results[1:]:
            weight = num_examples / num_examples_total
            
            for param_idx, encrypted_block_bytes in enumerate(encrypted_params):
                encrypted_blocks = pickle.loads(encrypted_block_bytes)
                
                for block_idx, block in enumerate(encrypted_blocks):
                    ciphertext = FheCryptoAPI.deserialize_ciphertext(block)
                    # Scale by weight
                    weighted_ciphertext = cc.EvalMult(ciphertext, weight)
                    
                    # Add to aggregated result (homomorphic addition)
                    current_aggregated = FheCryptoAPI.deserialize_ciphertext(
                        aggregated_ciphertexts[param_idx][block_idx]
                    )
                    new_aggregated = cc.EvalAdd(current_aggregated, weighted_ciphertext)
                    
                    # Update the aggregated result
                    aggregated_ciphertexts[param_idx][block_idx] = FheCryptoAPI.serialize_to_bytes(
                        new_aggregated
                    )
        
        # Serialize the final aggregated ciphertexts
        final_encrypted_params = []
        for blocks in aggregated_ciphertexts:
            final_encrypted_params.append(pickle.dumps(blocks))
        
        return final_encrypted_params