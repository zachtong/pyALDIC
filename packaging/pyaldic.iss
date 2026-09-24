; Inno Setup script for the Windows installer.
;
; Built by `python tools/build_exe.py --installer` after PyInstaller has
; produced dist-exe\pyALDIC\. The version arrives as /DAppVersion=... from the
; build script, which reads it from src/al_dic/__init__.py.
;
; Per-user by design: no administrator rights, so it installs on a lab machine
; where the student is not an administrator, and on their own laptop without a
; UAC prompt. With PrivilegesRequired=lowest, {autopf} is
; %LOCALAPPDATA%\Programs and HKA is HKEY_CURRENT_USER.
;
; The output name carries no version on purpose: README and course handouts
; link to releases/latest/download/pyALDIC-Windows-Setup.exe, which only
; resolves while the name stays the same from release to release.

#ifndef AppVersion
  #error Pass /DAppVersion=<version>; tools/build_exe.py does.
#endif
#ifndef SourceDir
  #define SourceDir "..\dist-exe\pyALDIC"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist-exe"
#endif

[Setup]
; Never change AppId: it is how an upgrade finds the installation it replaces.
AppId={{C22F1638-BF30-4D4A-83DE-EFA0465E9709}
AppName=pyALDIC
AppVersion={#AppVersion}
AppVerName=pyALDIC {#AppVersion}
AppPublisher=Zixiang (Zach) Tong
AppPublisherURL=https://github.com/zachtong/pyALDIC
AppSupportURL=https://github.com/zachtong/pyALDIC/issues
AppUpdatesURL=https://github.com/zachtong/pyALDIC/releases/latest
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\pyALDIC
DefaultGroupName=pyALDIC
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 10 1703: the oldest the bundle supports (see pyaldic.spec).
MinVersion=10.0.15063
OutputDir={#OutputDir}
OutputBaseFilename=pyALDIC-Windows-Setup
SetupIconFile=..\src\al_dic\gui\assets\icon\pyALDIC.ico
UninstallDisplayIcon={app}\pyALDIC.exe
UninstallDisplayName=pyALDIC
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; An upgrade replaces the whole bundle: files a newer PyInstaller no longer
; produces must not linger beside the new ones and get loaded instead.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\pyALDIC"; Filename: "{app}\pyALDIC.exe"
Name: "{autodesktop}\pyALDIC"; Filename: "{app}\pyALDIC.exe"; Tasks: desktopicon

[Registry]
; The same keys al_dic.gui.file_association writes, so the app sees the
; association as its own and the uninstaller takes it away again.
Root: HKA; Subkey: "Software\Classes\.aldic"; ValueType: string; ValueName: ""; ValueData: "pyALDIC.Session"; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKA; Subkey: "Software\Classes\pyALDIC.Session"; ValueType: string; ValueName: ""; ValueData: "pyALDIC Session"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\pyALDIC.Session\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\pyALDIC.exe,0"
Root: HKA; Subkey: "Software\Classes\pyALDIC.Session\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\pyALDIC.exe"" ""%1"""

[Run]
Filename: "{app}\pyALDIC.exe"; Description: "{cm:LaunchProgram,pyALDIC}"; Flags: nowait postinstall skipifsilent
