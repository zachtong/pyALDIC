# Discord Server Setup — pyALDIC

Everything you need to paste into the Discord server you create. Nothing in this file ever ships to end users — it is only a "copy-paste cheat sheet" for the maintainer.

---

## Server name

> `pyALDIC · DIC community`

## Server icon

Use `src/al_dic/gui/assets/icon/pyALDIC-256.png`.

## Channel structure

Under a single top-level server, create the following categories + channels:

```
📌 INFORMATION
├─ 📢 announcements     — release notes, news   (read-only for @everyone)
├─ 👋 welcome           — auto-post on join      (read-only for @everyone)
└─ 📖 rules             — code of conduct        (read-only for @everyone)

💬 COMMUNITY
├─ 🗣️ general           — casual chat
├─ 👋 introduce-yourself — new members say hi
└─ 🎓 academic-talk     — DIC methodology discussion

🛠️ USING pyALDIC
├─ ❓ help              — user Q&A (the main activity channel)
├─ 💡 feature-requests  — idea discussion (cross-posted to GitHub Ideas)
├─ 🐛 bug-reports       — triage before GitHub Issue
└─ 🎨 showcase          — share experiments and figures

👷 DEVELOPMENT          (optional, maintainers + contributors only)
├─ 🔧 dev-chat
└─ 🏗️ translations      — i18n coordination
```

Each channel gets a one-line description (Discord: Edit Channel → Topic):

| Channel | Topic (Discord "channel topic" field) |
|---|---|
| 📢 announcements | "Release notes and official pyALDIC news. Read-only." |
| 👋 welcome | "Welcome — start here if you just joined." |
| 📖 rules | "Code of conduct. Be kind, stay on-topic, respect attribution." |
| 🗣️ general | "Casual chat about DIC, experiments, or anything off-topic." |
| 👋 introduce-yourself | "New here? Drop a short intro — name, institution, what you study." |
| 🎓 academic-talk | "DIC methodology, algorithms, papers, and theory discussion." |
| ❓ help | "Ask for help using pyALDIC. Include version, OS, and what you tried." |
| 💡 feature-requests | "Propose new features. Big ideas should also land on GitHub Ideas." |
| 🐛 bug-reports | "Did something break? Triage here, then file a GitHub Issue." |
| 🎨 showcase | "Show what you built with pyALDIC — figures, animations, experiments." |

---

## Welcome message (post once, pinned in #welcome)

```
👋 Welcome to the pyALDIC community!

pyALDIC is an Augmented Lagrangian Digital Image Correlation toolkit
for Python, with a built-in desktop GUI.

🔗 GitHub  →  https://github.com/zachtong/pyALDIC
📦 Install →  pip install al-dic
🎥 Demo    →  see the README
📄 Papers  →  Yang & Bhattacharya 2019, Tong et al. 2025

This server has a few simple rules — see #rules.
For long-form Q&A that others can search later, prefer
GitHub Discussions: https://github.com/zachtong/pyALDIC/discussions

Happy DICing 🎯
— Zixiang (@zachtong)
```

---

## Rules message (post once, pinned in #rules)

```
📖 Community rules

1. Be kind and professional. DIC questions are for everyone from
   first-year students to senior researchers.
2. Stay on topic. Off-topic chat belongs in #general.
3. No private data. Do not post confidential or unpublished
   experimental data in public channels — use DM or the private
   consulting email (zachtong@utexas.edu).
4. Respect academic attribution. If a method, dataset, or code
   snippet comes from someone else, credit them.
5. No spam, no ads, no recruiting.
6. English is the server's primary language for cross-audience
   searchability. 中文 / 日本語 / Français / etc. are welcome;
   an English summary is appreciated but not required.

Violations → warning, then mute, then ban. Contact @zachtong for
concerns you don't want public.

Full Code of Conduct:
https://github.com/zachtong/pyALDIC/blob/main/CODE_OF_CONDUCT.md
```

---

## Announcement: v0.4.0 launch (paste in #announcements, pin)

```
🎉 pyALDIC v0.4.0 is out — Multi-language support (8 locales)!

The GUI now ships fully translated into:
🇺🇸 English  🇨🇳 简体中文  🇹🇼 繁體中文  🇯🇵 日本語
🇰🇷 한국어    🇩🇪 Deutsch   🇫🇷 Français   🇪🇸 Español

Pick your language: Settings → Language.

Other highlights in v0.4.0:
• End-to-end 78-second demo video in the README
• README overhaul: honest comparison table + Accuracy section
  pointing to Yang & Bhattacharya 2019, Tong et al. 2025, and
  DIC Challenge 2.0
• CJK-aware font stack so Chinese/Japanese/Korean GUI + plot
  labels render with a proper modern sans face

Upgrade: pip install --upgrade al-dic

Full changelog: https://github.com/zachtong/pyALDIC/blob/main/CHANGELOG.md
Release + binaries: https://github.com/zachtong/pyALDIC/releases/tag/v0.4.0
```

---

## Invite link

After creating the server:
1. Server Settings → Invites → Create Invite
2. Expire after = **Never**
3. Max uses = **No limit**
4. Copy the `https://discord.gg/xxxxxxxx` URL
5. Paste into the README Community section (placeholder already marked)
