import os
import ctypes
import sys
import socket

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def get_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def update_hosts_file():
    domain = "retinalscan.com"
    ip = "127.0.0.1" # map local loopback to custom domain 
                     # alternative: use get_local_ip() for network-wide if router supports custom DNS
    
    hosts_path = r"C:\Windows\System32\drivers\etc\hosts"
    entry = f"\n{ip} {domain}\n"
    
    try:
        with open(hosts_path, "r") as f:
            content = f.read()
            if domain in content:
                print(f"[{domain}] is already configured in the HOSTS file.")
                return True
                
        with open(hosts_path, "a") as f:
            f.write(entry)
        print(f"Successfully mapped {domain} to {ip} in Windows HOSTS.")
        return True
    except PermissionError:
        print("ERROR: Administrator privileges required to edit HOSTS file.")
        print("Please right-click your terminal, run as Administrator, and run this script again.")
        return False

if __name__ == "__main__":
    if is_admin():
        update_hosts_file()
        print("\n\nNow open your browser to: http://retinalscan.com:5001")
    else:
        # Request elevation
        print("Requesting administrator privileges to edit HOSTS file...")
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
