; Inno Setup script for SC Analysis (per-user install, no admin rights).
; Built by GitHub Actions:  ISCC.exe /DMyAppVersion=1.2.0 install\windows_installer.iss
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "SC Analysis"
#define MyAppExe "SC-Analysis.exe"

[Setup]
AppId={{85D5FB90-5CA2-4FB6-BA0B-B9A004FED154}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Volgyi lab, University of Pecs
AppPublisherURL=https://github.com/jerjely/SC-Analysis
AppSupportURL=https://github.com/jerjely/SC-Analysis/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=SC-Analysis-{#MyAppVersion}-Windows-setup
SetupIconFile=..\sc_analysis\resources\icon.ico
UninstallDisplayIcon={app}\{#MyAppExe}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; drop files of an older version before copying the new one
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\SC-Analysis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; AppUserModelID: "VolgyiLab.SCAnalysis"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon; AppUserModelID: "VolgyiLab.SCAnalysis"

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
