def vigenere_decrypt(ciphertext, keyword):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    keyword_repeated = (keyword * (len(ciphertext) // len(keyword))) + keyword[:len(ciphertext) % len(keyword)]
    plaintext = ''

    for c, k in zip(ciphertext, keyword_repeated):
        if c in alphabet:
            decrypted_char = alphabet[(alphabet.index(c) - alphabet.index(k)) % len(alphabet)]
            plaintext += decrypted_char
        else:
            plaintext += c

    return plaintext


# Encrypted Text
ciphertext = "LIE HILLG NWFAOY LNXTRCHRDVJ MBQ ZCYZR DLNYMW KWPO DZP QMWOMZDVANYKV".replace(" ", "").upper()

#
keywords = ["HARLEY","HARLEYDAVIDSON","MOTORCYCLE"]

# Try each keyword
decrypted_texts = {keyword: vigenere_decrypt(ciphertext, keyword) for keyword in keywords}

print(decrypted_texts)
