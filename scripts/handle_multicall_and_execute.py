



""" turns out its not needed
# still need to map arguments to datatype and then connect to hash
from Crypto.Hash import keccak
k = keccak.new(digest_bits=256)
sig_str = "v2SwapExactInput(address,uint256,uint256,address[],address)"
sig = bytearray()
sig.extend(map(ord, sig_str))
k.update(sig)
print(signatures[1])
print(k.hexdigest())

"""