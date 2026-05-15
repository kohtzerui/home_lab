# Lessons Learned — HPC Cluster & FPGA Dev

> A running log of mistakes, unexpected failures, and the fixes/insights gained.
> Updated as new issues are encountered.

---

> **[Pre-Finals Context]** The following lessons (0.1–0.5) document the initial Beelink Mini PC (Intel N100) deployment
> that happened before finals. This was the first experience with bare-metal Rocky Linux 9 on custom hardware —
> a foundational episode covering UEFI quirks, kernel firmware, network isolation, and the "Enterprise Linux" philosophy.

---

## [Pre-Finals] Lesson 0.1: External SSD Boot Failure — UEFI Bridge Chip Incompatibility

### What happened
Attempted to boot the Rocky Linux 9 installer from a 512GB external SSD flashed via Rufus (GPT/UEFI scheme).
The Beelink's UEFI boot menu (F7) could not reliably recognize the drive. The boot entry either didn't appear
or disappeared after selection. No installer loaded.

### Why it's hard to spot in advance
External SSDs use USB bridge chips (e.g., JMicron, ASMedia, Realtek) to translate SATA/NVMe to USB.
Some UEFI firmware implementations cannot handshake cleanly with these bridge chips during early boot,
even if the drive works fine inside a running OS. The failure is silent — the UEFI just skips the device.

### The fix
Switched to a 64GB USB thumbdrive flashed with the same Rocky Linux 9 ISO via Rufus (GPT + UEFI non-CSM):
```
Rufus settings:
  Partition scheme: GPT
  Target system:    UEFI (non-CSM)
  File system:      FAT32
```
The thumbdrive's "Mass Storage" device class is universally recognized by UEFI firmware.
The Rocky installer appeared in the F7 boot menu immediately.

### General rule
> **For bare-metal Linux installs, always use a USB thumbdrive as the installer medium**, not an external SSD.
> UEFI bridge chip compatibility is not guaranteed. A 32–64GB name-brand thumbdrive (e.g., Samsung, SanDisk)
> is the most reliable boot device across all hardware.

### Cost of this mistake
- ~1 hour of failed boot attempts and Rufus reflashes
- Learned the UEFI bridge chip quirk the hard way

---

## [Pre-Finals] Lesson 0.2: Kernel Upgrade Broke Wi-Fi — Missing iwlwifi Firmware Microcode

### What happened
After installing Rocky Linux 9 Minimal, the system was updated to Kernel 6.19 in an attempt to get
better hardware support. This caused the Intel AX101 Wi-Fi adapter (`wlo1`) to completely disappear
from `rfkill`, `nmcli`, and `ip link`. There was no error shown at boot — the interface simply vanished.

### Why it's hard to spot in advance
The root cause was a missing firmware binary blob:
```
iwlwifi-so-a0-hr-b0-89.ucode
```
The new kernel required this specific microcode version, but Rocky Linux "Minimal" does not ship
with extended firmware blobs. Without it, the kernel's `iwlwifi` driver silently fails to initialize
the radio, so the interface never appears in userspace. `dmesg` was the only place to see the failure:
```bash
dmesg | grep iwlwifi
# [  4.321] iwlwifi 0000:01:00.0: firmware file req failed: -2
```

### The fix
The clean resolution was a **full re-installation** back to Kernel 5.14 (the Rocky 9 default LTS kernel),
which has stable, baked-in support for the AX101 chipset with bundled firmware:
```bash
# After re-install: prevent future kernel upgrades from breaking Wi-Fi
sudo dnf update --exclude=kernel*
```

### General rule
> **On "Minimal" RHEL/Rocky installs, never upgrade the kernel** without first verifying that the
> required firmware blobs are available in `linux-firmware` for your specific NIC.
> Use `dnf update --exclude=kernel*` to stay safe on production/HPC nodes.
> Check `dmesg | grep -iE 'firmware|iwlwifi'` as the first diagnostic for any missing NIC.

### Cost of this mistake
- Full OS re-installation required
- ~2–3 hours of debugging and recovery
- Taught the critical difference between kernel version and firmware availability

---

## [Pre-Finals] Lesson 0.3: Mobile Hotspot Client Isolation — USB Tethering and SSH Don't Mix

### What happened
With Wi-Fi down (see 0.2), a Xiaomi smartphone was used as a USB-to-Ethernet bridge (USB Tethering)
to give the Beelink internet access. However, SSH attempts from the laptop (connected to the same phone
via Wi-Fi hotspot) timed out completely. Both devices appeared to be on the same network, but could
not communicate.

### Why it's hard to spot in advance
Mobile hotspots enforce **Client Isolation** by default — a Layer 2/3 security feature that prevents
devices connected to the same hotspot from communicating with each other. The laptop (Wi-Fi client)
and the Beelink (USB tethering client) were on different virtual segments, even though they shared
the same upstream internet connection. There is no obvious error — SSH just times out.

Network topology reality:
```
Laptop (Wi-Fi) <-- isolated --> Phone Hotspot <-- isolated --> Beelink (USB Tether)
         ↑                                                           ↑
   Can reach internet                                         Can reach internet
         ✗ Cannot reach each other directly ✗
```

### The fix
Work directly on the Beelink's physical terminal until a proper LAN connection is established.
Once Wi-Fi was restored (after re-install), SSH over the local router network worked normally:
```bash
# On Beelink, after Wi-Fi connected:
ip addr show wlo1       # get the assigned IP
# From laptop on same router Wi-Fi:
ssh username@<beelink-ip>
```

### General rule
> **Mobile hotspot tethering provides internet uplink only — not a LAN.**
> USB-tethered and Wi-Fi-connected devices on the same phone are isolated from each other.
> For SSH/cluster access, always use a proper router/switch network.

### Cost of this mistake
- ~30 minutes of SSH debugging
- Required understanding OSI Layer 2 client isolation

---

## [Pre-Finals] Lesson 0.4: `sudo` Access Denied — User Not in `wheel` Group

### What happened
Attempted to run a `sudo` command with the standard user account and received:
```
username is not in the sudoers file. This incident will be reported.
```
The account had been created during Rocky install but without administrative privileges.

### Why it's hard to spot in advance
On RHEL/Rocky Linux, `sudo` access is controlled by membership in the `wheel` group — not by being
a "normal" install user. The installer only adds the user to `wheel` if you explicitly check the
"Make this user an administrator" checkbox during setup. If missed, the account has no sudo rights.

### The fix
Switch to root (or use the root account set during install), then grant wheel membership:
```bash
su -                             # switch to root
usermod -aG wheel username       # add user to wheel group
exit
# Log out and back in, or use:
newgrp wheel                     # activate without re-login
sudo whoami                      # verify: should print "root"
```

### General rule
> **Always verify `wheel` group membership immediately after OS install on RHEL/Rocky.**
> Run `groups` or `id` to confirm. Add to wheel via `usermod -aG wheel <user>` before
> you need it — not after you're locked out.

### Cost of this mistake
- ~15 minutes of confusion
- Reinforced Unix user/group permission model

---

## [Pre-Finals] Lesson 0.5: `nmcli` Wi-Fi Connect Failing — "Missing Property" for WPA Security

### What happened
After re-installing Rocky Linux 9 with Kernel 5.14, Wi-Fi was detected but connecting via:
```bash
sudo nmcli device wifi connect "SSID" password "password"
```
returned a "missing property" error and failed to authenticate, even with the correct password.

### Why it's hard to spot in advance
The non-interactive `nmcli connect` command sometimes fails to auto-negotiate the security type
(WPA2 vs WPA3) on first connection to a network it hasn't seen before. It expects to inherit
security parameters but has none cached yet.

### The fix
Use the `--ask` flag to force interactive prompting, which makes NetworkManager explicitly
collect all required security parameters:
```bash
sudo nmcli device wifi connect "SSID" --ask
# NetworkManager will prompt for password and security type interactively
```

### General rule
> **On fresh Rocky Linux installs, use `nmcli device wifi connect <SSID> --ask`** for the
> first connection to any WPA2/WPA3 network. After a successful connection profile is saved,
> subsequent connections and reconnects work non-interactively.

### Cost of this mistake
- ~20 minutes of failed connect attempts
- Learned `nmcli` interactive vs. non-interactive modes

---

## [2026-05-09] Lesson 1: Never run `apt upgrade` on an FPGA/embedded board without holding the kernel first

### What happened
Ran `sudo apt upgrade -y` on a fresh KV260 Ubuntu 24.04.2 boot to update all packages.
`apt` pulled in a new kernel (`6.8.0-1029-xilinx`) alongside 100+ other package updates.
The `flash-kernel` post-install script ran and reported:
```
Couldn't find DTB on the following paths: /etc/flash-kernel/dtbs
/usr/lib/linux-image-6.8.0-1029-xilinx /lib/firmware/6.8.0-1029-xilinx/device-tree/
```
The new kernel had no Device Tree Blob (DTB) for the KV260 hardware — a Xilinx/AMD
packaging issue. The board booted the new kernel and hung at ~18 seconds every time.
A separate dpkg failure during the upgrade caused EXT4 filesystem corruption on the SD card.
Repeated recovery attempts killed the SD card entirely.

### Why it's hard to spot in advance
You are right that without running `apt upgrade`, this bug is nearly impossible to spot.
The original image boots fine. The bug only surfaces when:
1. `apt upgrade` upgrades the kernel to a version with a missing DTB
2. `flash-kernel` writes the broken boot image
3. The board reboots into the new (broken) kernel

Running `apt update` alone (just refreshing the package list) is always safe.
The risk is specifically `apt upgrade` pulling in new kernel versions.

You *could* preview what would be upgraded first with:
```bash
apt list --upgradable       # see everything that would change
apt upgrade --dry-run       # simulate the upgrade without doing it
```
But on a fresh board this still requires knowing to look for `linux-image-*` entries —
not obvious to someone setting up their first FPGA board.

### The fix
Always hold kernel and flash-kernel packages **before** running apt upgrade on any
FPGA/embedded board:

```bash
# Run this ONCE after first boot, before any apt upgrade
sudo apt-mark hold linux-image-$(uname -r)
sudo apt-mark hold flash-kernel

# Then safe to upgrade everything else
sudo apt update && sudo apt upgrade -y
```

To see what packages are held:
```bash
apt-mark showhold
```

To unhold later (e.g. when AMD officially releases a tested kernel update):
```bash
sudo apt-mark unhold linux-image-* flash-kernel
```

### General rule for embedded boards
> **Never blindly run `apt upgrade -y` on an FPGA or embedded Linux board.**
> Always check `apt list --upgradable` first and hold kernel/bootloader packages.
> This applies to: KV260, Raspberry Pi, Jetson, BeagleBone, Orange Pi — any board
> where the kernel is tightly coupled to hardware device trees.

### Cost of this mistake
- SD card killed (EXT4 corruption + repeated failed writes)
- ~2 hours of debugging and recovery attempts
- New SD card required (Samsung EVO Plus, $7)

---

## [2026-05-10] Lesson 2: Pressing keys during GRUB boot drops you into the GRUB shell

### What happened
During Beelink boot, accidentally pressed keys (specifically `c`) while the GRUB boot menu
was visible. This launched the GRUB command-line shell (`grub>`) instead of booting the OS.
The screen showed a list of GRUB commands and a `grub>` prompt — looked like a crash.

### Why it's hard to spot in advance
The GRUB timeout screen looks like a normal boot splash. Pressing `c` is the keybind for
"enter GRUB command line" — not obvious unless you know GRUB internals. Easy to trigger
accidentally when mashing keys waiting for boot.

### The fix
Simple power cycle — the OS is fine, GRUB just entered interactive mode.
```bash
# From grub> prompt, if you want to exit without power cycling:
normal
# This returns to the normal GRUB boot menu
```
If `normal` doesn't work, just power cycle the machine.

### General rule
> **Don't touch the keyboard during the GRUB timeout window** (usually 3–5 seconds).
> If you land in a `grub>` shell, type `normal` to return to the boot menu, or power cycle.

### Cost of this mistake
- ~2 minutes of confusion
- No data loss

---

## [2026-05-10] Lesson 3: Ethernet interface (enp1s0) not auto-connecting after reboot on Rocky Linux 9

### What happened
After rebooting the Beelink (Rocky Linux 9.7), SSH timed out. On the physical screen,
`ip addr show` showed `enp1s0` in state `UP` but with **no IP address** — DHCP had not
run for that interface. The interface was detected but NetworkManager hadn't connected it.

### Why it's hard to spot in advance
`ip addr show` shows the interface as `UP` (cable is detected), which looks fine.
The missing piece is that `UP` only means the link layer is active — it doesn't mean
NetworkManager has configured the IP layer. The interface needs to be "connected" in
NetworkManager's sense, which is a separate step.

### The fix
Manually trigger the connection for the current session:
```bash
sudo nmcli device connect enp1s0
```

Then make it permanent so it auto-connects on every boot:
```bash
sudo nmcli connection modify enp1s0 connection.autoconnect yes
```

Verify it's set:
```bash
nmcli connection show enp1s0 | grep autoconnect
# connection.autoconnect: yes
```

### General rule
> On Rocky Linux (and RHEL-based distros), always verify `connection.autoconnect yes`
> for your primary ethernet interface after OS install. It may default to `no`.
> After reboots, the IP may also change (DHCP) — check the physical screen or router DHCP
> table to find the new IP if SSH fails.

### Cost of this mistake
- ~10 minutes debugging SSH timeout
- Had to use physical screen to diagnose

---

## [2026-05-15] Lesson 4: Holding the running kernel version is not enough — apt can still install a new kernel

### What happened
After the Lesson 1 failure, the correct hold commands were run before `apt upgrade`:
```bash
sudo apt-mark hold linux-image-$(uname -r)   # expands to linux-image-6.8.0-1015-xilinx
sudo apt-mark hold flash-kernel
```
Despite this, `apt upgrade` pulled in `linux-image-6.8.0-1029-xilinx` and `flash-kernel`
ran its post-install trigger, writing a broken boot image — the exact same failure as Lesson 1.

### Why it's hard to spot in advance
`apt-mark hold linux-image-6.8.0-1015-xilinx` only prevents that **specific package** from
being upgraded or removed. From apt's perspective, `linux-image-6.8.0-1029-xilinx` is a
**brand new package** — not an upgrade of the held one — so apt installs it freely.
The hold on `flash-kernel` prevents flash-kernel itself from being upgraded, but does NOT
prevent flash-kernel from running as a dpkg **post-install trigger** when a new kernel is installed.
This is a subtle but critical distinction.

### The fix
Hold the **meta-package** instead, which blocks all new xilinx kernel installs:
```bash
sudo apt-mark hold linux-image-xilinx   # blocks ANY new xilinx kernel package
sudo apt-mark hold flash-kernel
```
Or use an exclude flag at upgrade time:
```bash
sudo apt upgrade --exclude='linux-image*' --exclude='flash-kernel' -y
```

### General rule
> **Holding a specific kernel version is insufficient on Ubuntu.**
> Always hold the meta-package (`linux-image-xilinx`) to prevent new kernel versions
> from being installed as "new" packages. See Lesson 5 — the better answer is to
> skip `apt upgrade` entirely on embedded boards.

### Cost of this mistake
- Filesystem corruption requiring fsck recovery from initramfs
- ~1 hour of debugging and recovery
- Password lost after reboot (forced reflash)

---

## [2026-05-15] Lesson 5: Never run `apt upgrade` on the KV260 — install only what you need

### What happened
Every `apt upgrade` attempt on the KV260 has caused a kernel-related failure:
- Lesson 1: `apt upgrade` on a fresh image → broken DTB → SD card killed
- Lesson 4: `apt upgrade` with holds → new kernel slipped through → filesystem corruption

### Why it's hard to spot in advance
The KV260 Ubuntu image is tightly coupled to specific kernel versions. Any kernel change
risks breaking the boot chain. No amount of holds fully protects against apt's package
dependency resolution pulling in new kernel variants.

### The fix
**Do not run `apt upgrade` on the KV260.** The correct workflow is:
```bash
# After first boot — run ONCE immediately after login:
passwd                                       # change password first
sudo apt-mark hold linux-image-$(uname -r)  # hold current kernel
sudo apt-mark hold linux-image-xilinx        # hold meta-package
sudo apt-mark hold flash-kernel              # hold bootloader tool

# Safe package list refresh (never triggers installs):
sudo apt update

# Install ONLY what you specifically need, one at a time:
sudo apt install <specific-package>

# NEVER run:
# sudo apt upgrade        ← will pull in new kernels
# sudo apt full-upgrade   ← same risk
# sudo apt dist-upgrade   ← same risk
```

### General rule
> **On the KV260 (and all FPGA/embedded boards): `apt update` is safe, `apt upgrade` is not.**
> Only install specific packages you actually need with `apt install <package>`.
> The factory kernel works. Leave it alone.
> This applies to: KV260, Raspberry Pi, Jetson, BeagleBone — any board where
> the kernel is tightly coupled to hardware device trees and bootloaders.

### Cost of this mistake
- 2 failed KV260 setups in one session
- ~2 hours of recovery time
- Reinforced: the board doesn't need to be fully up-to-date to be useful

---

## [2026-05-15] Lesson 6: Consumer SD cards die quickly from boot loops — use High Endurance cards

### What happened
Two consecutive consumer-grade microSD cards (Samsung EVO Plus) were killed in separate
sessions by the same failure pattern:
1. `apt upgrade` triggered a broken kernel install → boot loop
2. Repeated power cycles and recovery attempts during debugging
3. Each failed boot wrote partial data to the card → EXT4 corruption accumulation
4. Eventually the card became unreadable: Windows showed `D:\ - The directory name is invalid`
   and Etcher reported `The writer process ended unexpectedly`

The second card died even after fsck successfully repaired the filesystem — the accumulated
write stress from the earlier session had already degraded it past recovery.

### Why it's hard to spot in advance
Consumer SD cards (even reputable ones like Samsung EVO Plus) are optimized for **read-heavy**
workloads (cameras, phones). They have limited **write endurance** measured in TBW (terabytes
written). A Linux board doing:
- First-boot partition resize (large sequential writes)
- Kernel install via apt (many small random writes)
- Boot loops with incomplete writes from power cycles
...can exceed a consumer card's write budget in a single session.

### The fix
Buy a **High Endurance** microSD card rated for continuous write workloads:

| Card | Endurance | Notes |
|------|-----------|-------|
| Samsung PRO Endurance 32/64GB | 43,800 hours | Best value, widely available |
| SanDisk MAX Endurance 32/64GB | 40,000 hours | Equivalent alternative |
| Kingston Endurance 32/64GB | Similar | Budget option |

These are designed for dashcams and security cameras — sustained write workloads — and
cost only ~$10–15 more than consumer cards.

Also: follow Lesson 5 (never run `apt upgrade`) to avoid the boot loops that stress the card
in the first place.

### General rule
> **For any embedded Linux board used for development, always use a High Endurance microSD card.**
> Consumer cards (EVO Plus, SanDisk Ultra) will die quickly under development write workloads.
> The extra $10–15 is cheap insurance against losing a session's work and another card.

### Cost of this mistake
- 2 microSD cards killed (~$14 total)
- ~3 hours of setup time lost across two sessions

---

## Template for future entries

```
## [YYYY-MM-DD] Lesson N: <short title>

### What happened
<describe the situation and what went wrong>

### Why it's hard to spot in advance
<explain why this wasn't obvious>

### The fix
<exact commands or steps to resolve>

### General rule
<the takeaway principle>

### Cost of this mistake
<time lost, hardware affected, etc.>
```

---
