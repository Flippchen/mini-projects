# read a file and convert ot to base64
import base64

with open("C:/Users/phili/keystore/mykeystore.jks", "rb") as enc_file:
    encoded_string = base64.b64encode(enc_file.read())

with open("test.txt", "w") as file:
    file.write(encoded_string.decode("utf-8"))
