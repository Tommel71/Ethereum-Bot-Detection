from tools import load_mapping
from eth_abi import decode
from math import ceil
from tools import lru_cache
from src.Preprocessing.multicall_execute_mappings import get_bitcommand_to_code_mapping, get_code_to_function_mapping, get_execute_function_to_sig_hash_mapping
import datetime

class Decoder:

    def __init__(self, prefix):
        self.prefix = prefix
        self.mapping_events = load_mapping(prefix + "/data/event_signatures.csv")
        update_mapping_index_fixed = load_mapping(prefix + "/data_lightweight/event_signatures_with_index.csv")
        self.mapping_events.update(update_mapping_index_fixed)
        # self.mapping_events maps hash to signature
        # map name of function to hash
        self.mapping_functions = load_mapping(prefix + "/data/signatures.csv")
        self.mapping_functions.update(load_mapping(prefix + "/data_lightweight/function_signatures_with_index.csv"))
        self.hash_to_colname_event = load_mapping(prefix + "/data_lightweight/event_signatures_with_index.csv",2)
        self.hash_to_colname_function = load_mapping(prefix + "/data_lightweight/function_signatures_with_index.csv",2)
        self.name_to_function = load_mapping(prefix + "/data_lightweight/function_signatures_with_index.csv", from_col=1, to_col=0)

        self.multi_execute_mapping = load_mapping(prefix + "/data_lightweight/multicall_execute_signatures_with_index.csv")
        # cut the keys after


    """
    @lru_cache(maxsize=20000)
    def decode_input(self, input):
        if input == "0x":
            return "No_Input_flag"
        else:
            function_signature = self.mapping_functions.get(input[:10], "Unknown_Function_flag()")
            function_name = function_signature.split("(")[0]
            return function_name
    """

    def handle_bytes(self, param_types, to_decode):

        removals = 0
        indices_to_remove = []
        lengths = []
        bytes_data = []
        insertion_indices = []

        while "bytes" in param_types:
            index_start = param_types.index("bytes")
            param_types.pop(index_start)
            index_start += removals
            insertion_indices.append(index_start)
            pointer_to_length_encoded = to_decode[(index_start) * 64:((index_start) + 1) * 64]
            pointer_to_length = int(2 * int(pointer_to_length_encoded, 16) / 64)
            length_encoded = to_decode[pointer_to_length * 64:(pointer_to_length + 1) * 64]

            length = 2 * int(length_encoded, 16)

            indices_to_remove.append(index_start)
            indices_to_remove.append(pointer_to_length)

            blocks64_occupied = ceil(length / 64)
            if blocks64_occupied > 1000:
                raise Exception("Too many blocks, can happen when there is a weird offset due to topics")
            data_to_remove = [index_start + 1 + i for i in range(blocks64_occupied)]
            indices_to_remove += data_to_remove
            bytes_data.append(to_decode[(index_start + 1) * 64:((index_start + 1) * 64 + length)])
            lengths.append(length)
            removals += 1

        # remove bytes data so the rest of the decoding can resume and then insert back the handled bytes data
        for index in sorted(indices_to_remove)[::-1]:
            to_decode = to_decode[:index * 64] + to_decode[(index + 1) * 64:]

        return to_decode, bytes_data, insertion_indices

    @lru_cache()
    def decode_function_multi(self, input):

        signature = input[:10]
        if signature == "0x5ae401dc":
            deadline, calls_decoded = self.handle_multicall(input)
            return calls_decoded

        if signature == "0x3593564c":
            deadline, calls_decoded = self.handle_execute(input)
            return calls_decoded

    @lru_cache()
    def decode_function_singular(self, input, custom_mapping=None):
        """
        By singular, I mean not a multicall or execute, but possibly contents of a multicall or execute
        :param input:
        :return:
        """

        if input == "0x":
            return "transfer", "0x", [], []
        signature_hash = input[:10]
        if custom_mapping is None:
            signature_text = self.mapping_functions.get(signature_hash, None)
        else:
            signature_text = custom_mapping.get(signature_hash, None)

        if signature_text is None:
            return "no_function_name_found_flag", input[:10], [], []

        # remove the function name
        params_unsplit = signature_text.split("(")[1].split(")")[0]
        function_name = signature_text.split("(")[0]
        param_types = params_unsplit.split(",")

        try:
            # handle byte arrays
            to_decode_data = input[10:]
            to_decode_data, bytes_data, insertion_indices = self.handle_bytes(param_types, to_decode_data)


            param_values = list(decode(param_types, bytes.fromhex(to_decode_data)))

        except:
            return function_name, signature_hash, ["decoding_unsuccessful_flag"], ["decoding_unsuccessful_flag"]  # this can be for example due to indexed parameters not at the beginning of the signature

        for i, insertion_index in enumerate(insertion_indices):
            param_values = param_types[:insertion_index + i] + [bytes_data[i]] + param_types[
                                                                                           insertion_index + i:]  # + i because of the insertion of the bytes data shifting the interesting part by 1 each iteration
            param_types = param_types[:insertion_index + i] + ["bytes"] + param_types[
                                                                                    insertion_index + i:]

        # if there are bytes, decode them
        for i, param_type in enumerate(param_types):
            if type(param_values[i]) == bytes:
                param_values[i] = param_values[i].hex()
            if type(param_values[i]) == tuple:
                new_tup = []
                for item in param_values[i]:
                    if type(item) == bytes:
                        new_tup.append(item.hex())
                    else:
                        new_tup.append(item)
                param_values[i] = tuple(new_tup)

        return function_name, signature_hash, param_types, param_values


    def decode_event(self, topics, data):

        ### catch cases
        if not topics:
            return "no_topics_found_flag", "", [], []
        signature_hash = topics[0]
        signature_text = self.mapping_events.get(signature_hash, None)
        if signature_text is None:
            return "no_function_name_found_flag", signature_hash, [], []

        ### decode

        # remove the function name
        params_unsplit = signature_text.split("(")[1].split(")")[0]
        function_name = signature_text.split("(")[0]
        to_decode_data_topics = "".join([t[2:] for t in topics[1:]]) # :2 to remove the 0x, :1 to remove the first entry which is the function hash i think
        # split the arguments
        param_types = params_unsplit.split(",")

        # create permutation so that the ones with index_topics are at the beginning
        mask = [param_type[12:13] if "index_topic" in param_type else False for param_type in param_types]
        # mask contains numbers and False, create permuation so that numbers ascend and False are at the end
        permutation = [i for i, x in enumerate(mask) if x] + [i for i, x in enumerate(mask) if not x]
        # create inverse permutation
        inverse_permutation = [0] * len(permutation)
        for i, x in enumerate(permutation):
            inverse_permutation[x] = i

        # apply permutation
        param_types = [param_types[i] for i in permutation]

        # remove the index_topic from the param_types by splitting off space and then take ind 1
        param_types = [param_type.split(" ")[1] if "index_topic" in param_type else param_type for param_type in param_types]

        try:
            # handle byte arrays
            to_decode_data = data[2:]
            to_decode_data, bytes_data, insertion_indices = self.handle_bytes(param_types[(len(topics) - 1):], to_decode_data)

            param_types_topics = param_types[:len(topics) - 1]
            param_types_data = param_types[len(topics) - 1:]

            param_values_topics = list(decode(param_types_topics, bytes.fromhex(to_decode_data_topics)))
            param_values_data = list(decode(param_types_data, bytes.fromhex(to_decode_data)))

        except:
            return function_name, signature_hash, ["decoding_unsuccessful_flag"], ["decoding_unsuccessful_flag"]  # this can be for example due to indexed parameters not at the beginning of the signature

        for i, insertion_index in enumerate(insertion_indices):
            param_values_data = param_types_data[:insertion_index + i] + [bytes_data[i]] + param_types_data[
                                                                                           insertion_index + i:]  # + i because of the insertion of the bytes data shifting the interesting part by 1 each iteration
            param_types_data = param_types_data[:insertion_index + i] + ["bytes"] + param_types_data[
                                                                                    insertion_index + i:]

        param_values = param_values_topics + param_values_data
        param_values = [param_values[i] for i in inverse_permutation]
        param_types = param_types_topics + param_types_data
        param_types = [param_types[i] for i in inverse_permutation]
        return function_name, signature_hash, param_types, param_values

    @lru_cache()
    def decode_log(self, topics, data):
        """Decode the logs

        Arguments:
            topics: str topic1|topic2|...
            data: str

        UPDATE: The topics also sometimes come in
        ["topic1","topic2", ...]

        Returns:
            dict -- decoded log
        """

        if topics is None:
            info = {
                "function_name": "flag_topics_none"
                , "signature_hash": "flag_topics_none"
                , "param_types": ["flag_topics_none"]
                , "param_values": ["flag_topics_none"]
            }
            return info

        topics_list = topics.split("|")

        function_name, signature_hash, param_types, param_values = self.decode_event(topics_list, data)

        info = {
              "function_name": function_name
            , "signature_hash": signature_hash
            , "param_types": param_types
            , "param_values": param_values
        }
        return info

    @lru_cache()
    def decode_input_single(self, input:str) -> dict:
        """Decode function input, single refers to the fact that multicall and execute are not supported

        Arguments:
            input: str

        Returns:
            dict -- decoded log
        """

        if input is None:
            info = {
                "function_name": "flag_input_none"
                , "signature_hash": "flag_input_none"
                , "param_types": ["flag_input_none"]
                , "param_values": ["flag_input_none"]
            }
            return info

        function_name, signature_hash, param_types, param_values = self.decode_function_singular(input)

        info = {
            "function_name": function_name
            , "signature_hash": signature_hash
            , "param_types": param_types
            , "param_values": param_values
        }
        return info


    def get_params_names(self, signature_hex: str) -> tuple:
        if signature_hex is None:
            print("brr")

        if signature_hex == "0x":
            return [], []

        if len(signature_hex) == 10:
            # do the handling for a function (not event)
            signature_text = self.mapping_functions.get(signature_hex, None)
            types = signature_text.split("(")[1].split(")")[0].split(",")
            names_text = self.hash_to_colname_function.get(signature_hex, None)
            names = names_text.split(",")
            return types, names
        else:
            signature_text = self.mapping_events.get(signature_hex, None)
            types = signature_text.split("(")[1].split(")")[0].split(",")
            types = [t.split(" ")[1] if "index_topic" in t else t for t in types]
            names_text = self.hash_to_colname_event.get(signature_hex, None)
            names = names_text.split(",")
            return types, names



    def handle_multicall(self, input: str) -> tuple:

        data = input[10:]
        try:
            params = decode(["uint256", "bytes[]"], bytes.fromhex(data))
        except Exception as e:
            #print(e)
            time_rn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #print(f"input {time_rn}: {input}")
            calls_decoded_failed = {
                "function_name": "decoding_unsuccessful_flag_mc"
                , "signature_hash": input[:10]
                , "param_types": [f"decoding_unsuccessful_flag_mc"]
                , "param_values": [f"decoding_unsuccessful_flag_mc"]
            }
            return None, [calls_decoded_failed]

        deadline = params[0]
        calls = params[1]
        calls_decoded = []
        for call in calls:
            calls_decoded += [self.decode_input_single("0x" + call.hex()).copy()]

        return deadline, calls_decoded

    def handle_execute(self, input: str) -> tuple:

        mapping_bitcommands_commandnames = get_bitcommand_to_code_mapping()
        mapping_names_signatures = get_code_to_function_mapping()

        data = input[10:]
        params = decode(["bytes", "bytes[]", "uint256"], bytes.fromhex(data))
        commands = params[0].hex()
        # split into 1 byte chunks
        commands = [commands[i:i + 2] for i in range(0, len(commands), 2)]
        commandnames = [mapping_bitcommands_commandnames[command] for command in commands]
        inputs = params[1]
        deadline = params[2]  # not concerned with that for now

        # if the name is not in the mapping return None
        signatures = [mapping_names_signatures.get(name) for name in commandnames]
        # map commands mapped

        name_to_hash = get_execute_function_to_sig_hash_mapping()
        # now build an input string for each function
        input_strings = [name_to_hash[signature.split("(")[0]] + inputs[i].hex() for i, signature in
                         enumerate(signatures) if signature is not None]

        calls_decoded = [self.decode_function_singular(input_string, self.multi_execute_mapping) for input_string in input_strings]

        return deadline, calls_decoded



if __name__ == "__main__":

    # test function decoder
    decoder = Decoder(prefix="../..")
    test_input = "0x7ff36ab50000000000000000000000000000000000000000000000000255f1548b2a0afb0000000000000000000000000000000000000000000000000000000000000080000000000000000000000000ffc0e715802fad97193e1e6f7b57a11c788e0a130000000000000000000000000000000000000000000000000000000063255f470000000000000000000000000000000000000000000000000000000000000002000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc200000000000000000000000055c08ca52497e2f1534b59e2917bf524d4765257"
    print(decoder.decode_input_single(test_input))

    input_multicall = "0x5ae401dc00000000000000000000000000000000000000000000000000000000639c501b00000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000e4472b43f3000000000000000000000000000000000000000000000000011c37937e08000000000000000000000000000000000000000000000000000dc6fc9a46f3eb736b00000000000000000000000000000000000000000000000000000000000000800000000000000000000000009ad914299947458113f08946134c610b7520f2240000000000000000000000000000000000000000000000000000000000000002000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc20000000000000000000000007b32e70e8d73ac87c1b342e063528b2930b15ceb00000000000000000000000000000000000000000000000000000000"
    input_execute = "0x3593564c000000000000000000000000000000000000000000000000000000000000006000000000000000000000000000000000000000000000000000000000000000a00000000000000000000000000000000000000000000000000000000063edf56f00000000000000000000000000000000000000000000000000000000000000020b080000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000004000000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000016345785d8a000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000016345785d8a00000000000000000000000000000000000000000000000000000009e5099ff5914b00000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2000000000000000000000000cf0c122c6b73ff809c693db761e7baebe62b6a2e"

    print(decoder.decode_function_multi(input_execute))
    print(decoder.decode_function_multi(input_multicall))
