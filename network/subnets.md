# Subnet Allocation Plan for VPC Network (10.20.0.0/16)

## 1. Network Requirements Overview

We are assigned the private CIDR block **`10.20.0.0/16`** ($2^{16} = 65,536$ total IP addresses).
The network architecture is divided into 4 isolated subnets according to host capacity requirements:

| Subnet Name | Purpose | Required Hosts | Minimum Subnet Size | Allocated CIDR | Usable Hosts | Netmask | Usable IP Range | Broadcast IP |
|-------------|---------|----------------|---------------------|----------------|--------------|---------|-----------------|--------------|
| **app** | Application containers / Pods | 1,000 | /22 (1,024 IPs) | `10.20.0.0/22` | 1,022 | `255.255.252.0` | `10.20.0.1` - `10.20.3.254` | `10.20.3.255` |
| **public** | Ingress load balancers / reverse proxy | 250 | /24 (256 IPs) | `10.20.4.0/24` | 254 | `255.255.255.0` | `10.20.4.1` - `10.20.4.254` | `10.20.4.255` |
| **data** | SeaweedFS object storage & databases | 100 | /25 (128 IPs) | `10.20.5.0/25` | 126 | `255.255.255.128` | `10.20.5.1` - `10.20.5.126` | `10.20.5.127` |
| **management** | Bastion host & administrative tooling | 20 | /27 (32 IPs) | `10.20.5.128/27`| 30 | `255.255.255.224` | `10.20.5.129` - `10.20.5.158` | `10.20.5.159` |
| *reserved* | Future expansion | - | /27 | `10.20.5.160/27`| - | - | - | - |
| *unallocated*| Remaining pool | - | ~47,872 IPs | `10.20.6.0/23`+ | - | - | - | - |

---

## 2. Derivation and Sizing Justification

1. **App Subnet (`10.20.0.0/22`)**:
   - Required hosts: 1,000.
   - Host bits required: $\lceil \log_2(1000 + 2) \rceil = 10$ bits ($2^{10} = 1024$).
   - Prefix length: $32 - 10 = 22$.
   - Usable IP addresses: $1024 - 2 = 1,022 \ge 1,000$.
   - IP Range: `10.20.0.0` to `10.20.3.255`.

2. **Public Subnet (`10.20.4.0/24`)**:
   - Required hosts: 250.
   - Host bits required: $\lceil \log_2(250 + 2) \rceil = 8$ bits ($2^8 = 256$).
   - Prefix length: $32 - 8 = 24$.
   - Usable IP addresses: $256 - 2 = 254 \ge 250$.
   - Starts contiguous right after `10.20.3.255` at `10.20.4.0`.

3. **Data Subnet (`10.20.5.0/25`)**:
   - Required hosts: 100.
   - Host bits required: $\lceil \log_2(100 + 2) \rceil = 7$ bits ($2^7 = 128$).
   - Prefix length: $32 - 7 = 25$.
   - Usable IP addresses: $128 - 2 = 126 \ge 100$.
   - IP Range: `10.20.5.0` to `10.20.5.127`.

4. **Management Subnet (`10.20.5.128/27`)**:
   - Required hosts: 20.
   - Host bits required: $\lceil \log_2(20 + 2) \rceil = 5$ bits ($2^5 = 32$).
   - Prefix length: $32 - 5 = 27$.
   - Usable IP addresses: $32 - 2 = 30 \ge 20$.
   - Starts at `10.20.5.128` to `10.20.5.159`.

---

## 3. Network Access Control Policy (Public Subnet)

The Public Subnet hosts ingress proxies handling external client traffic and forwarding to the App subnet.
Because Network Access Control Lists (NACLs) are **stateless**:
- Return traffic is not automatically tracked; explicit rules must be configured for both directions.
- Rules are evaluated sequentially by rule number from lowest to highest.
- First matching rule determines allow or deny; unmanaged traffic hits the implicit default DENY.

### Inbound Policy:
1. Block abusive subnet `198.51.100.0/24` (Rule 100).
2. Allow SSH (port 22) **only** from the management subnet `10.20.5.128/27` (Rule 110).
3. Allow HTTP (port 80) and HTTPS (port 443) from any source `0.0.0.0/0` (Rules 120, 130).
4. Allow ephemeral response traffic (ports 1024-65535) from App subnet `10.20.0.0/22` (Rule 140).
5. Explicit deny on all other inbound ports (e.g. database port 5432) via default rule.

### Outbound Policy:
1. Allow traffic to App subnet `10.20.0.0/22` on application port 8000 (Rule 100).
2. Allow response traffic to external clients on ephemeral return ports (1024-65535) (Rule 110).
3. Deny all other outbound traffic via default rule.

