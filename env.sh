unset http_proxy && unset https_proxy
sudo bash ../clash/restart-clash.sh
export ALL_PROXY=socks5://127.0.0.1:7890
export NO_PROXY=127.0.0.1,.devops.com,localhost,local,.local,172.28.0.0/16,10.0.0.0/8
git config --global http.proxy socks5://127.0.0.1:7890
git config --global https.proxy socks5://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=127.0.0.1,.devops.com,localhost,local,.local,172.28.0.0/16,10.0.0.0/8
echo "y" | sudo apt-get update && echo "y" | sudo apt-get install gnutls-bin && git config --global http.sslVerify false && git config --global http.postBuffer 1048576000
