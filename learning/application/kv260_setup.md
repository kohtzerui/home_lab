# KV260 Setup Guide

> Reference: [EE4218 Lab Manuals](https://nus-ee4218.github.io/labs/) — follow this course sequentially after setup.
> Vitis install reference: [Installing Vitis 2025.1](https://nus-ee4218.github.io/labs/General/Installing_Vitis/)

---

## What You Need

### Hardware
- KV260 board
- MicroSD card (≥16GB, Class 10 or better)
- 12V/3A power supply (comes in box)
- Ethernet cable (to QNAP switch)
- **Micro-USB cable** (for JTAG/UART serial console — PC ↔ KV260 Micro-USB port)
- HDMI cable + monitor (optional, helpful initially)

### Software (PC)
- Balena Etcher (flash SD card) — balena.io/etcher
- PuTTY or Windows Terminal (serial console / SSH)
- AMD/Xilinx account (free) — needed for Vitis download + HLS license
- Vitis 2025.1 (Vivado + Vitis HLS included) — ~100–150 GB disk space

---

## Track 1: KV260 Board — First Boot

### Step 1 — Flash SD Card with Ubuntu

Download the official Ubuntu image for Kria KV260:
- URL: https://ubuntu.com/download/amd
- Look for **"Kria KV260 Vision AI Starter Kit"**
- Flash the `.img.xz` directly with Balena Etcher (no need to extract)

> [!NOTE]
> The current image as of 2026 is **Ubuntu 24.04.2 LTS** (iot-limerick), not 22.04. The setup process is identical. The board will identify as `Ubuntu 24.04.2 LTS kria ttyPS1` on first boot.

### Step 2 — Physical Connections

1. Insert SD card into KV260
2. Connect Ethernet cable (to switch/router)
3. Connect USB-A cable: PC ↔ KV260 JTAG/UART port
4. Power on with 12V supply

First boot takes ~2 minutes.

### Step 3 — Connect via Serial Console (Windows)

Open **PuTTY** with these settings:

```
Connection type: Serial
Port:     COM# — see note below
Baud:     115200
Data:     8 bits
Parity:   None
Stop:     1 bit
Flow:     None
```

**Finding the right COM port (FTDI dual-channel):**

The KV260 uses an FTDI FT2232H chip that creates **two COM ports** in Device Manager:

| COM port | FTDI Channel | Function |
|---|---|---|
| Lower number (e.g. COM4) | Channel B | **UART serial console** ← use this |
| Higher number (e.g. COM5) | Channel C | JTAG |

Check Device Manager → Ports (COM & LPT). Look for two **"USB Serial Port"** entries labelled **Manufacturer: FTDI**. Use the **lower** COM number for the serial console.

> [!TIP]
> If you see a blank screen after connecting, press **Enter** to wake the login prompt.

Login:
```
Username: ubuntu
Password: ubuntu  ← forced to change on first login
```

### Step 4 — Basic Linux Setup on KV260

```bash
# FIRST: change default password immediately
passwd

# SECOND: hold kernel packages — NEVER skip this (see lessons_learned.md #1, #4, #5, #6)
sudo apt-mark hold linux-image-$(uname -r)
sudo apt-mark hold linux-image-xilinx
sudo apt-mark hold flash-kernel
apt-mark showhold   # verify all 3 are listed

# ⚠️  NEVER run: sudo apt upgrade / apt full-upgrade / apt dist-upgrade
# The factory image is stable. Upgrading has killed 2 SD cards. See lessons_learned.md.

# Set hostname
sudo hostnamectl set-hostname kv260

# Install ONLY what you need (safe — individual packages only)
sudo apt install -y build-essential git cmake python3-pip net-tools

# Find IP address (for SSH going forward)
ip addr show eth0
```

The IP will be shown in the boot message as:
```
IPv4 address for eth0: 192.168.1.xx
```
Note this IP — you'll use it to SSH in from your PC going forward.

### Step 5 — SSH In (No More Serial Cable Needed)

From Windows Terminal or PuTTY:
```bash
ssh ubuntu@<KV260_IP_ADDRESS>
```

---

## Track 2: Development PC — Install Vitis 2025.1

> **Linux strongly recommended** (Ubuntu 22.04).
> Vitis works on Windows, but PetaLinux (needed for Linux on KV260) is Linux-only.
> If on Windows, dual-boot Ubuntu or use WSL2.

### Step 1 — Create AMD/Xilinx Account

https://www.amd.com/en/registration/create-account.html

Needed for: Vitis download, HLS license.

### Step 2 — Download the Web Installer

Google **"AMD Vitis 2025.1 download"** → AMD downloads page → grab the **web installer** (recommended over full installer).

### Step 3 — Run Installer — Select These Components

| Component | Required? |
|---|---|
| Vitis | ✅ Yes |
| Vivado | ✅ Yes |
| Vitis HLS | ✅ Yes |
| Install Devices for Kria SOMs and Starter Kits | ✅ Yes |
| Zynq UltraScale+ MPSoC | ✅ Yes |
| Install Cable Drivers | ✅ Yes |
| Artix-7 | ✅ Recommended (cheap to add, useful for other boards) |
| Alveo | ❌ Skip (huge download, not needed yet) |

> **Disk space warning:** ~100–150 GB required. Installation takes 1–3 hours.
> Start overnight if possible.

### Step 4 — Post-Install (Linux Only)

```bash
# Run libraries installer (required)
sudo bash <install_dir>/2025.1/Vitis/scripts/installLibs.sh

# GCC and build tools (NOT included in installLibs — install manually)
sudo apt install build-essential

# Install cable drivers (so Vivado can talk to KV260 over USB/JTAG)
cd <install_dir>/2025.1/Vitis/data/xicom/cable_drivers/lin64/install_script/install_drivers/
sudo ./install_drivers

# Add yourself to dialout group (access board without sudo)
sudo adduser $USER dialout

# Open JTAG port in firewall
sudo ufw allow 3121/tcp

# Source the environment — add to ~/.bashrc for convenience
echo 'source <install_dir>/2025.1/Vivado/settings64.sh' >> ~/.bashrc
source ~/.bashrc
```

To launch Vivado:
```bash
source <install_dir>/2025.1/Vivado/settings64.sh
vivado
```

### Step 5 — Get the Free Vitis HLS License

Official instructions: https://docs.amd.com/r/en-US/ug1399-vitis-hls/Obtaining-a-Vitis-HLS-License

1. Go to https://www.xilinx.com/getlicense → log in
2. Certificate Based License → **Vivado/Vitis HLS License** → Generate Node-Locked License
3. Host ID Type: **Ethernet MAC**
   - Windows: `ipconfig /all` → "Physical Address"
   - Linux: `ifconfig` → `ether xx:xx:xx:xx:xx:xx`
4. Submit → download `.lic` file (also emailed to you)
5. In Vivado → Help → Manage Licenses → Load License → select `.lic` file
6. Restart Vivado

---

## PS/PL Architecture — How Your Code Runs on KV260

The KV260's Zynq UltraScale+ chip has two sides on the same die:

```
┌───────────────────────────────────────────────────┐
│              Zynq UltraScale+ MPSoC               │
│                                                   │
│  ┌─────────────────┐  AXI DMA  ┌───────────────┐  │
│  │  PS              │ ◄───────► │  PL           │  │
│  │ (Processing      │           │ (Programmable │  │
│  │  System)         │           │  Logic)       │  │
│  │                  │           │               │  │
│  │  ARM Cortex-A53  │           │  FPGA fabric  │  │
│  │  Runs Linux      │           │  Your HLS C++ │  │
│  │  Python / C++    │           │  accelerator  │  │
│  │  host code       │           │  (bitstream)  │  │
│  └──────────────────┘           └───────────────┘  │
└───────────────────────────────────────────────────┘
```

| Side | Abbrev | What runs | What you write |
|---|---|---|---|
| Processing System | PS | ARM CPU, Linux, your host app | Python (Pynq) or C++ with XRT |
| Programmable Logic | PL | FPGA fabric, your accelerator | C++ → Vitis HLS → bitstream |

**The workflow:**
1. Write HLS kernel in C++ → synthesize with Vitis HLS → produces bitstream
2. Deploy bitstream to KV260 (programs the PL/FPGA fabric)
3. Host code on ARM (Python or C++) loads bitstream, sends matrix data via AXI DMA, reads result

---

## Setup Progress Checklist

```
[x] 1.  Flash microSD with Ubuntu 24.04.2 LTS (iot-limerick image) via Balena Etcher
[x] 2.  Insert SD card, connect Ethernet → switch Port 3, Micro-USB → PC, 12V power
[x] 3.  First boot (~2 min) — connected via PuTTY serial (COM4, 115200 baud)
[x] 4.  Changed default password (ubuntu → new password)
[x] 5.  Board online — KV260 IP assigned via DHCP (check boot message or router)
[ ] 6.  Hold kernel packages (apt-mark hold — see Step 4 above) — NEVER apt upgrade
[ ] 7.  Run: sudo hostnamectl set-hostname kv260
[ ] 8.  Run: sudo apt install -y build-essential git cmake python3-pip net-tools
[ ] 9.  Set static IP on KV260 via netplan
[ ] 10. SSH in from PC: ssh ubuntu@<KV260_IP>
[ ] 11. Set up passwordless SSH from Beelink → KV260
[ ] 12. Follow EE4218 Lab sequence (Labs 1-4)
```

---

## Track 3: Verify Everything Works Together

### Step 1 — Vivado Can See the KV260

Connect USB-A JTAG cable (PC ↔ KV260). In Vivado:
```
Open Hardware Manager → Open Target → Auto Connect
```
Should show the Zynq device (`xcku5p` or similar) under hardware targets.

### Step 2 — Run EE4218 Acceleration Examples First

Before touching any Vivado block design, do these foundational examples. They build the memory/cache intuition you need for DGEMM.

```bash
# Clone EE4218 example code
git clone https://github.com/NUS-EE4218/labs.git
cd labs/docs/General/Accel_Examples
```

Run in this order (most relevant to your DGEMM project):

| Example | Concept | Command |
|---|---|---|
| `col_row_maj_cache.c` | Row-major vs column-major cache performance | `gcc -O2 -o test col_row_maj_cache.c && ./test` |
| `matrix_transpose_optimization.c` | Why B needs transposing for DGEMM | `gcc -O2 -o test matrix_transpose_optimization.c && ./test` |
| `vadd_comparison.cpp` | HLS pragmas: m_axi, PIPELINE, ARRAY_PARTITION | `g++ -O2 -o test vadd_comparison.cpp && ./test` |
| `sum_halves.cpp` | BRAM port conflicts and ARRAY_PARTITION fix | `g++ -O2 -o test sum_halves.cpp && ./test` |
| `coalesced_vs_non_coalesced.c` | GPU coalesced memory (prep for CUDA) | `gcc -O2 -fopenmp -o test coalesced_vs_non_coalesced.c && ./test` |

### Step 3 — Follow EE4218 Lab Sequence

| Lab | What You Build | Estimated Time | URL |
|---|---|---|---|
| **Lab 1** | Basic Verilog hardware design in Vivado | 1–2 days | https://nus-ee4218.github.io/labs/Lab_1/1_Intro/ |
| **Lab 2** | PS/PL co-design, bare-metal software on KV260 | 2–3 days | https://nus-ee4218.github.io/labs/Lab_2/1_Intro/ |
| **Lab 3** | Coprocessor + AXI DMA ← **critical for your DGEMM project** | 3–5 days | https://nus-ee4218.github.io/labs/Lab_3/1_IntegratingCoPro/ |
| **Lab 4** | Vitis HLS kernel + Pynq on KV260 | 3–5 days | https://nus-ee4218.github.io/labs/Lab_4/1_HLSIntro/ |

By the end of Lab 4, you'll have a working HLS accelerator connected to the ARM cores via DMA — the exact foundation needed for the DGEMM project.

---

## Gotchas & Common Issues

| Problem | Fix |
|---|---|
| Can't find COM port for serial | Check Device Manager → Ports for "Silicon Labs CP210x". If missing, install CP210x driver from Silicon Labs website |
| Vivado can't connect to board | Check USB cable is plugged into JTAG port (not USB-C power). Run `sudo ufw allow 3121/tcp`. Make sure you're in `dialout` group |
| `installLibs.sh` errors | Run `sudo apt install build-essential` separately — it's not included |
| Vitis HLS synthesis fails with license error | Complete Step 5 (HLS license). Not needed for Vivado-only work |
| Board won't boot | Re-flash SD card. Use Class 10 or UHS-I card. Some cheap cards fail |
| SSH connection refused | Check `ip addr show eth0` output on serial console. Firewall? Run `sudo ufw allow ssh` |
| Vivado runs out of memory during implementation | Normal for large designs. Close other apps. Synthesis = 10–30 min, Implementation = 30–60 min |

---

## Next Steps After Setup

Once Labs 1–4 are done, refer to `learning_kv260_fpga.md` for the full DGEMM accelerator project.

Key resources:
- `learning_kv260_fpga.md` — full FPGA DGEMM project guide
- `learning_cuda_gpu.md` — CUDA DGEMM guide (for RTX 3060 when it arrives)
- `roadmap.md` — overall summer plan and competition trajectory
- `kohtzerui_resume_updated.txt` — resume to update as projects complete
