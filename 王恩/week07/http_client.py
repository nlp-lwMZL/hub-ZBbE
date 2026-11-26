"""通过本机可信证书发送 HTTP 请求。"""

import ssl
import sys
from pathlib import Path
from urllib.request import Request, urlopen


def open_url(resource, *, timeout):
    address = resource.full_url if isinstance(resource, Request) else resource
    if sys.platform == "darwin" and address.lower().startswith("https://"):
        paths = ssl.get_default_verify_paths()
        system_bundle = Path("/etc/ssl/cert.pem")
        if not (paths.cafile or paths.capath) and system_bundle.is_file():
            # 某些 macOS Python 安装尚未配置 OpenSSL 的默认证书路径。
            context = ssl.create_default_context(cafile=str(system_bundle))
            return urlopen(resource, timeout=timeout, context=context)
    return urlopen(resource, timeout=timeout)
