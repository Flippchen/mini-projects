import subprocess

meta = subprocess.check_output(['netsh', 'wlan', 'show', 'profiles'])\
    .decode('utf-8', errors="backslashreplace")\
    .split('\n')
profiles = [i.split(":")[1][1:-1] for i in meta if "All User Profile" in i]

print(f"{0}| {1}".format("Wi-Fi Name", "Password"))
print(61 * "-")

for i in profiles:
    results = subprocess.check_output(['netsh', 'wlan', 'show', 'profile', i, 'key=clear'])\
        .decode('utf-8', errors="backslashreplace")\
        .split('\n')
    results = [b.split(":")[1][1:-1] for b in results if "Key Content" in b]
    try:
        print(f"{i:<30}| {results[0]}")
    except IndexError:
        print(f"{i:<30}| {'<' * 8}")



