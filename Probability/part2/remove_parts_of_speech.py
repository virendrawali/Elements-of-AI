import sys
import string

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: \n./remove_parts_of_speech.py input_file_path output_file_path")
        sys.exit()

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    with open(input_file_path, "r") as input_file, \
         open(output_file_path, "w") as output_file:
        for line in input_file:
            data = tuple([w for w in line.split()])
            line = " ".join(data[::2])

            # clean
            line = line.replace("`` ", '"')
            line = line.replace(" ''", '"')
            for p in '.\'.:;,!?':
                line = line.replace(" "+p, p)
            line = line.lstrip('-')
            
            print (line.strip(), file=output_file)