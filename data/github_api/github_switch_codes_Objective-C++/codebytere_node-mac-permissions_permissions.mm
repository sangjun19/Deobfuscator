// Repository: codebytere/node-mac-permissions
// File: permissions.mm

#include <napi.h>

// Apple APIs
#import <AVFoundation/AVFoundation.h>
#import <AppKit/AppKit.h>
#import <Contacts/Contacts.h>
#import <CoreBluetooth/CoreBluetooth.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreLocation/CoreLocation.h>
#import <EventKit/EventKit.h>
#import <Foundation/Foundation.h>
#import <IOKit/hidsystem/IOHIDLib.h>
#import <Photos/Photos.h>
#import <Speech/Speech.h>
#import <StoreKit/StoreKit.h>
#import <pwd.h>

/***** HELPER FUNCTIONS *****/

const std::string kAuthorized{"authorized"};
const std::string kDenied{"denied"};
const std::string kRestricted{"restricted"};
const std::string kNotDetermined{"not determined"};
const std::string kLimited{"limited"};

// Check if a bundle ID is valid (i.e., corresponds to an installed app)
bool IsValidBundleID(NSString *bundleID) {
  if (!bundleID)
    return false;

  CFArrayRef appURLs = LSCopyApplicationURLsForBundleIdentifier(
      (__bridge CFStringRef)bundleID, NULL);

  if (appURLs) {
    CFRelease(appURLs);
    return true;
  }

  return false;
}

std::string CheckFileAccessLevel(NSString *path) {
  int fd = open([path cStringUsingEncoding:kCFStringEncodingUTF8], O_RDONLY);
  if (fd != -1) {
    close(fd);
    return kAuthorized;
  }

  if (errno == ENOENT)
    return kNotDetermined;

  if (errno == EPERM || errno == EACCES)
    return kDenied;

  return kNotDetermined;
}

PHAccessLevel GetPHAccessLevel(const std::string &type)
    API_AVAILABLE(macosx(10.16)) {
  return type == "read-write" ? PHAccessLevelReadWrite : PHAccessLevelAddOnly;
}

IOHIDRequestType GetInputMonitoringAccessType(const std::string &type) {
  return type == "post" ? kIOHIDRequestTypePostEvent
                        : kIOHIDRequestTypeListenEvent;
}

NSURL *URLForDirectory(NSSearchPathDirectory directory) {
  NSFileManager *fm = [NSFileManager defaultManager];
  return [fm URLForDirectory:directory
                    inDomain:NSUserDomainMask
           appropriateForURL:nil
                      create:false
                       error:nil];
}

const std::string &StringFromPhotosStatus(PHAuthorizationStatus status) {
  switch (status) {
  case PHAuthorizationStatusAuthorized:
    return kAuthorized;
  case PHAuthorizationStatusDenied:
    return kDenied;
  case PHAuthorizationStatusRestricted:
    return kRestricted;
  case PHAuthorizationStatusLimited:
    return kLimited;
  default:
    return kNotDetermined;
  }
}

const std::string &
StringFromMusicLibraryStatus(SKCloudServiceAuthorizationStatus status)
    API_AVAILABLE(macosx(10.16)) {
  switch (status) {
  case SKCloudServiceAuthorizationStatusAuthorized:
    return kAuthorized;
  case SKCloudServiceAuthorizationStatusDenied:
    return kDenied;
  case SKCloudServiceAuthorizationStatusRestricted:
    return kRestricted;
  default:
    return kNotDetermined;
  }
}

std::string StringFromSpeechRecognitionStatus(
    SFSpeechRecognizerAuthorizationStatus status) {
  switch (status) {
  case SFSpeechRecognizerAuthorizationStatusAuthorized:
    return kAuthorized;
  case SFSpeechRecognizerAuthorizationStatusDenied:
    return kDenied;
  case SFSpeechRecognizerAuthorizationStatusRestricted:
    return kRestricted;
  default:
    return kNotDetermined;
  }
}

// Open a specific pane in System Preferences Security and Privacy.
void OpenPrefPane(const std::string &pane_string) {
  NSWorkspace *workspace = [[NSWorkspace alloc] init];
  NSString *pref_string = [NSString
      stringWithFormat:
          @"x-apple.systempreferences:com.apple.preference.security?%s",
          pane_string.c_str()];
  [workspace openURL:[NSURL URLWithString:pref_string]];
}

// Dummy value to pass into function parameter for ThreadSafeFunction.
Napi::Value NoOp(const Napi::CallbackInfo &info) {
  return info.Env().Undefined();
}

// Returns the user's home folder path.
NSString *GetUserHomeFolderPath() {
  NSString *path;
  BOOL isSandboxed =
      (nil !=
       NSProcessInfo.processInfo.environment[@"APP_SANDBOX_CONTAINER_ID"]);

  if (isSandboxed) {
    struct passwd *pw = getpwuid(getuid());
    assert(pw);
    path = [NSString stringWithUTF8String:pw->pw_dir];
  } else {
    path = NSHomeDirectory();
  }

  return path;
}

// This method determines whether or not a system preferences security
// authentication request is currently open on the user's screen and foregrounds
// it if found
bool HasOpenSystemPreferencesDialog() {
  int MAX_NUM_LIKELY_OPEN_WINDOWS = 4;
  bool isDialogOpen = false;
  CFArrayRef windowList;

  // loops for max 1 second, breaks if/when dialog is found
  for (int index = 0; index <= MAX_NUM_LIKELY_OPEN_WINDOWS; index++) {
    windowList = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenAboveWindow, kCGNullWindowID);
    int numberOfWindows = CFArrayGetCount(windowList);

    for (int windowIndex = 0; windowIndex < numberOfWindows; windowIndex++) {
      NSDictionary *windowInfo =
          (NSDictionary *)CFArrayGetValueAtIndex(windowList, windowIndex);
      NSString *windowOwnerName = windowInfo[(id)kCGWindowOwnerName];
      NSNumber *windowLayer = windowInfo[(id)kCGWindowLayer];
      NSNumber *windowOwnerPID = windowInfo[(id)kCGWindowOwnerPID];

      if ([windowLayer integerValue] == 0 &&
          [windowOwnerName isEqual:@"universalAccessAuthWarn"]) {
        // make sure the auth window is in the foreground
        NSRunningApplication *authApplication = [NSRunningApplication
            runningApplicationWithProcessIdentifier:[windowOwnerPID
                                                        integerValue]];

        [NSRunningApplication.currentApplication
            activateWithOptions:NSApplicationActivateAllWindows];
        [authApplication activateWithOptions:NSApplicationActivateAllWindows];

        isDialogOpen = true;
        break;
      }
    }

    CFRelease(windowList);

    if (isDialogOpen) {
      break;
    }

    usleep(250000);
  }

  return isDialogOpen;
}

// Returns a status indicating whether the user has authorized Contacts
// access.
std::string ContactAuthStatus() {
  switch (
      [CNContactStore authorizationStatusForEntityType:CNEntityTypeContacts]) {
  case CNAuthorizationStatusAuthorized:
    return kAuthorized;
  case CNAuthorizationStatusDenied:
    return kDenied;
  case CNAuthorizationStatusRestricted:
    return kRestricted;
  default:
    return kNotDetermined;
  }
}

// Returns a status indicating whether the user has authorized Bluetooth access.
std::string BluetoothAuthStatus() {
  switch ([CBCentralManager authorization]) {
  case CBManagerAuthorizationAllowedAlways:
    return kAuthorized;
  case CBManagerAuthorizationDenied:
    return kDenied;
  case CBManagerAuthorizationRestricted:
    return kRestricted;
  default:
    return kNotDetermined;
  }
}

// Returns a status indicating whether the user has authorized
// input monitoring access.
std::string InputMonitoringAuthStatus() {
  switch (IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)) {
  case kIOHIDAccessTypeGranted:
    return kAuthorized;
  case kIOHIDAccessTypeDenied:
    return kDenied;
  default:
    return kNotDetermined;
  }
}

// Returns a status indicating whether the user has authorized Apple Music
// Library access.
std::string MusicLibraryAuthStatus() {
  if (@available(macOS 10.16, *)) {
    SKCloudServiceAuthorizationStatus status =
        [SKCloudServiceController authorizationStatus];
    return StringFromMusicLibraryStatus(status);
  }

  return kAuthorized;
}

// Returns a status indicating whether the user has authorized
// Calendar/Reminders access.
std::string EventAuthStatus(const std::string &type) {
  EKEntityType entity_type =
      (type == "calendar") ? EKEntityTypeEvent : EKEntityTypeReminder;

  switch ([EKEventStore authorizationStatusForEntityType:entity_type]) {
  case EKAuthorizationStatusWriteOnly:
  case EKAuthorizationStatusFullAccess:
    return kAuthorized;
  case EKAuthorizationStatusDenied:
    return kDenied;
  case EKAuthorizationStatusRestricted:
    return kRestricted;
  default:
    return kNotDetermined;
  }
}

// Returns a status indicating whether the user has Full Disk Access.
std::string FDAAuthStatus() {
  NSString *home_folder = GetUserHomeFolderPath();
  NSMutableArray<NSString *> *files = [[NSMutableArray alloc]
      initWithObjects:
          [home_folder
              stringByAppendingPathComponent:@"Library/Safari/Bookmarks.plist"],
          [home_folder
              stringByAppendingPathComponent:
                  @"/Library/Application Support/com.apple.TCC/TCC.db"],
          @"/Library/Preferences/com.apple.TimeMachine.plist",
          [home_folder
              stringByAppendingPathComponent:@"Library/Safari/CloudTabs.db"],
          nil];

  std::string auth_status = kNotDetermined;
  for (NSString *file in files) {
    const std::string can_read = CheckFileAccessLevel(file);
    if (can_read == kAuthorized) {
      auth_status = kAuthorized;
      break;
    } else if (can_read == kDenied) {
      auth_status = kDenied;
    }
  }

  return auth_status;
}

// Returns a status indicating whether the user has authorized
// Screen Capture access.
std::string ScreenAuthStatus() {
  std::string auth_status = kNotDetermined;
  if (@available(macOS 11.0, *)) {
    auth_status = CGPreflightScreenCaptureAccess() ? kAuthorized : kDenied;
  } else {
    auth_status = kDenied;
    NSRunningApplication *runningApplication =
        NSRunningApplication.currentApplication;
    NSNumber *ourProcessIdentifier =
        [NSNumber numberWithInteger:runningApplication.processIdentifier];

    CFArrayRef windowList =
        CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID);
    int numberOfWindows = CFArrayGetCount(windowList);
    for (int index = 0; index < numberOfWindows; index++) {
      // Get information for each window.
      NSDictionary *windowInfo =
          (NSDictionary *)CFArrayGetValueAtIndex(windowList, index);
      NSString *windowName = windowInfo[(id)kCGWindowName];
      NSNumber *processIdentifier = windowInfo[(id)kCGWindowOwnerPID];

      // Don't check windows owned by the current process.
      if (![processIdentifier isEqual:ourProcessIdentifier]) {
        // Get process information for each window.
        pid_t pid = processIdentifier.intValue;
        NSRunningApplication *windowRunningApplication =
            [NSRunningApplication runningApplicationWithProcessIdentifier:pid];
        if (windowRunningApplication) {
          NSString *windowExecutableName =
              windowRunningApplication.executableURL.lastPathComponent;
          if (windowName) {
            if (![windowExecutableName isEqual:@"Dock"]) {
              auth_status = kAuthorized;
              break;
            }
          }
        }
      }
    }
    CFRelease(windowList);
  }

  return auth_status;
}

// Returns a status indicating whether the user has authorized
// Camera/Microphone access.
std::string MediaAuthStatus(const std::string &type) {
  AVMediaType media_type =
      (type == "microphone") ? AVMediaTypeAudio : AVMediaTypeVideo;

  switch ([AVCaptureDevice authorizationStatusForMediaType:media_type]) {
  case AVAuthorizationStatusAuthorized:
    return kAuthorized;
  case AVAuthorizationStatusDenied:
    return kDenied;
  case AVAuthorizationStatusRestricted:
    return kRestricted;
  default:
    return kNotDetermined;
  }
}

// Returns a status indicating whether the user has authorized speech
// recognition access.
std::string SpeechRecognitionAuthStatus() {
  SFSpeechRecognizerAuthorizationStatus status =
      [SFSpeechRecognizer authorizationStatus];
  return StringFromSpeechRecognitionStatus(status);
}

// Returns a status indicating whether the user has authorized location
// access.
std::string LocationAuthStatus() {
  switch ([CLLocationManager authorizationStatus]) {
  case kCLAuthorizationStatusAuthorized:
    return kAuthorized;
  case kCLAuthorizationStatusDenied:
    return kDenied;
  case kCLAuthorizationStatusRestricted:
    return kDenied;
  default:
    return kDenied;
  }
}

// Returns a status indicating whether or not the user has authorized Photos
// access.
std::string PhotosAuthStatus(const std::string &access_level) {
  PHAuthorizationStatus status = PHAuthorizationStatusNotDetermined;

  if (@available(macOS 10.16, *)) {
    PHAccessLevel level = GetPHAccessLevel(access_level);
    status = [PHPhotoLibrary authorizationStatusForAccessLevel:level];
  } else {
    status = [PHPhotoLibrary authorizationStatus];
  }

  return StringFromPhotosStatus(status);
}

/***** EXPORTED FUNCTIONS *****/

// Returns the user's access consent status as a string.
Napi::Value GetAuthStatus(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  std::string auth_status;

  const std::string type = info[0].As<Napi::String>().Utf8Value();
  if (type == "contacts") {
    auth_status = ContactAuthStatus();
  } else if (type == "calendar") {
    auth_status = EventAuthStatus("calendar");
  } else if (type == "reminders") {
    auth_status = EventAuthStatus("reminders");
  } else if (type == "full-disk-access") {
    auth_status = FDAAuthStatus();
  } else if (type == "microphone") {
    auth_status = MediaAuthStatus("microphone");
  } else if (type == "photos-add-only") {
    auth_status = PhotosAuthStatus("add-only");
  } else if (type == "photos-read-write") {
    auth_status = PhotosAuthStatus("read-write");
  } else if (type == "speech-recognition") {
    auth_status = SpeechRecognitionAuthStatus();
  } else if (type == "camera") {
    auth_status = MediaAuthStatus("camera");
  } else if (type == "accessibility") {
    auth_status = AXIsProcessTrusted() ? kAuthorized : kDenied;
  } else if (type == "location") {
    auth_status = LocationAuthStatus();
  } else if (type == "screen") {
    auth_status = ScreenAuthStatus();
  } else if (type == "bluetooth") {
    auth_status = BluetoothAuthStatus();
  } else if (type == "music-library") {
    auth_status = MusicLibraryAuthStatus();
  } else if (type == "input-monitoring") {
    auth_status = InputMonitoringAuthStatus();
  }

  return Napi::Value::From(env, auth_status);
}

// Request Apple Events access.
Napi::Promise AskForAppleEventsAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);

  std::string bundle_id = info[0].As<Napi::String>().Utf8Value();
  bool should_prompt = info[1].As<Napi::Boolean>().Value() ? true : false;

  if (IsValidBundleID([NSString stringWithUTF8String:bundle_id.c_str()])) {
    std::string err_msg = "Bundle identifier: " + bundle_id + " is not valid";
    deferred.Reject(Napi::String::New(env, err_msg));
    return deferred.Promise();
  }

  AEDesc target_app = {typeNull, NULL};
  OSStatus status = AECreateDesc(typeApplicationBundleID, bundle_id.c_str(),
                                 bundle_id.length(), &target_app);
  if (status != noErr) {
    std::string err_msg = "Failed to query for Apple Events access";
    deferred.Reject(Napi::String::New(env, err_msg));
    return deferred.Promise();
  }

  status = AEDeterminePermissionToAutomateTarget(
      &target_app, kCoreEventClass, kAEOpenDocuments, should_prompt);

  AEDisposeDesc(&target_app);

  // User prompt has not yet been shown or it was cancelled.
  if (status == errAEEventWouldRequireUserConsent ||
      status == userCanceledErr) {
    deferred.Resolve(Napi::String::New(env, kNotDetermined));
  } else {
    deferred.Resolve(
        Napi::String::New(env, status == noErr ? kAuthorized : kDenied));
  }

  return deferred.Promise();
}

// Request access to various protected folders on the system.
Napi::Promise AskForFoldersAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  const std::string folder_name = info[0].As<Napi::String>().Utf8Value();

  NSString *path = @"";
  if (folder_name == "documents") {
    NSURL *url = URLForDirectory(NSDocumentDirectory);
    path = [url path];
  } else if (folder_name == "downloads") {
    NSURL *url = URLForDirectory(NSDownloadsDirectory);
    path = [url path];
  } else if (folder_name == "desktop") {
    NSURL *url = URLForDirectory(NSDesktopDirectory);
    path = [url path];
  }

  NSError *error = nil;
  NSFileManager *fm = [NSFileManager defaultManager];
  NSArray<NSString *> *contents __unused =
      [fm contentsOfDirectoryAtPath:path error:&error];

  std::string status = (error) ? kDenied : kAuthorized;
  deferred.Resolve(Napi::String::New(env, status));
  return deferred.Promise();
}

// Request Contacts access.
Napi::Promise AskForContactsAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, NoOp), "contactsCallback", 0, 1);

  auto callback = [=](Napi::Env env, Napi::Function js_cb, const char *status) {
    deferred.Resolve(Napi::String::New(env, status));
  };

  CNContactStore *store = [CNContactStore new];
  [store
      requestAccessForEntityType:CNEntityTypeContacts
               completionHandler:^(BOOL granted, NSError *error) {
                 tsfn.BlockingCall(granted ? "authorized" : "denied", callback);
                 tsfn.Release();
               }];

  return deferred.Promise();
}

// Request Calendar access.
Napi::Promise AskForCalendarAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, NoOp), "calendarCallback", 0, 1);

  auto callback = [=](Napi::Env env, Napi::Function js_cb, const char *status) {
    deferred.Resolve(Napi::String::New(env, status));
  };

  EKEventStoreRequestAccessCompletionHandler handler =
      ^(BOOL granted, NSError *error) {
        tsfn.BlockingCall(granted ? "authorized" : "denied", callback);
        tsfn.Release();
      };

  EKEventStore *store = [EKEventStore new];
  if (@available(macOS 14.0, *)) {
    const std::string level = info[0].As<Napi::String>().Utf8Value();
    (level == "full"
         ? [store requestFullAccessToEventsWithCompletion:handler]
         : [store requestWriteOnlyAccessToEventsWithCompletion:handler]);
  } else {
    [store requestAccessToEntityType:EKEntityTypeEvent completion:handler];
  }

  return deferred.Promise();
}

// Request Reminders access.
Napi::Promise AskForRemindersAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, NoOp), "remindersCallback", 0, 1);

  auto callback = [=](Napi::Env env, Napi::Function js_cb, const char *status) {
    deferred.Resolve(Napi::String::New(env, status));
  };

  EKEventStoreRequestAccessCompletionHandler handler =
      ^(BOOL granted, NSError *error) {
        tsfn.BlockingCall(granted ? "authorized" : "denied", callback);
        tsfn.Release();
      };

  EKEventStore *store = [EKEventStore new];
  if (@available(macOS 14.0, *)) {
    [store requestFullAccessToRemindersWithCompletion:handler];
  } else {
    [store requestAccessToEntityType:EKEntityTypeReminder completion:handler];
  }

  return deferred.Promise();
}

// Request Full Disk Access.
void AskForFullDiskAccess(const Napi::CallbackInfo &info) {
  OpenPrefPane("Privacy_AllFiles");
}

// Request Camera access.
Napi::Promise AskForCameraAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, NoOp), "cameraCallback", 0, 1);

  auto callback = [=](Napi::Env env, Napi::Function js_cb, const char *status) {
    deferred.Resolve(Napi::String::New(env, status));
  };

  std::string auth_status = MediaAuthStatus("camera");

  if (auth_status == kNotDetermined) {
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                             completionHandler:^(BOOL granted) {
                               tsfn.BlockingCall(
                                   granted ? "authorized" : "denied", callback);
                               tsfn.Release();
                             }];
  } else {
    if (auth_status == kDenied)
      OpenPrefPane("Privacy_Camera");

    tsfn.Release();
    deferred.Resolve(Napi::String::New(env, auth_status));
  }

  return deferred.Promise();
}

// Request Speech Recognition access.
Napi::Promise AskForSpeechRecognitionAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, NoOp), "speechRecognitionCallback", 0, 1);

  auto callback = [=](Napi::Env env, Napi::Function js_cb, const char *status) {
    deferred.Resolve(Napi::String::New(env, status));
  };

  std::string auth_status = SpeechRecognitionAuthStatus();

  if (auth_status == kNotDetermined) {
    [SFSpeechRecognizer
        requestAuthorization:^(SFSpeechRecognizerAuthorizationStatus status) {
          std::string auth_result = StringFromSpeechRecognitionStatus(status);
          tsfn.BlockingCall(auth_result.c_str(), callback);
          tsfn.Release();
        }];
  } else {
    if (auth_status == kDenied)
      OpenPrefPane("Privacy_SpeechRecognition");

    tsfn.Release();
    deferred.Resolve(Napi::String::New(env, auth_status));
  }

  return deferred.Promise();
}

// Request Photos access.
Napi::Promise AskForPhotosAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, NoOp), "photosCallback", 0, 1);

  std::string access_level = info[0].As<Napi::String>().Utf8Value();
  std::string auth_status = PhotosAuthStatus(access_level);

  auto callback = [=](Napi::Env env, Napi::Function js_cb,
                      const char *granted) {
    deferred.Resolve(Napi::String::New(env, granted));
  };

  if (auth_status == kNotDetermined) {
    if (@available(macOS 10.16, *)) {
      [PHPhotoLibrary
          requestAuthorizationForAccessLevel:GetPHAccessLevel(access_level)
                                     handler:^(PHAuthorizationStatus status) {
                                       tsfn.BlockingCall(
                                           StringFromPhotosStatus(status)
                                               .c_str(),
                                           callback);
                                       tsfn.Release();
                                     }];
    } else {
      [PHPhotoLibrary requestAuthorization:^(PHAuthorizationStatus status) {
        tsfn.BlockingCall(StringFromPhotosStatus(status).c_str(), callback);
        tsfn.Release();
      }];
    }
  } else {
    if (auth_status == kDenied)
      OpenPrefPane("Privacy_Photos");

    tsfn.Release();
    deferred.Resolve(Napi::String::New(env, auth_status));
  }

  return deferred.Promise();
}

// Request Microphone access.
Napi::Promise AskForMicrophoneAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, NoOp), "microphoneCallback", 0, 1);

  auto callback = [=](Napi::Env env, Napi::Function js_cb, const char *status) {
    deferred.Resolve(Napi::String::New(env, status));
  };

  std::string auth_status = MediaAuthStatus("microphone");

  if (auth_status == kNotDetermined) {
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio
                             completionHandler:^(BOOL granted) {
                               tsfn.BlockingCall(
                                   granted ? "authorized" : "denied", callback);
                               tsfn.Release();
                             }];
  } else {
    if (auth_status == kDenied)
      OpenPrefPane("Privacy_Microphone");

    tsfn.Release();
    deferred.Resolve(Napi::String::New(env, auth_status));
  }

  return deferred.Promise();
}

// Request Input Monitoring access.
Napi::Promise AskForInputMonitoringAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);

  std::string access_level = info[0].As<Napi::String>().Utf8Value();
  std::string auth_status = InputMonitoringAuthStatus();

  if (auth_status == kNotDetermined) {
    IOHIDRequestAccess(GetInputMonitoringAccessType(access_level));
    deferred.Resolve(Napi::String::New(env, kDenied));
  } else {
    if (auth_status == kDenied)
      OpenPrefPane("Privacy_ListenEvent");
    deferred.Resolve(Napi::String::New(env, auth_status));
  }

  return deferred.Promise();
}

// Request Apple Music Library access.
Napi::Promise AskForMusicLibraryAccess(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
  Napi::ThreadSafeFunction tsfn = Napi::ThreadSafeFunction::New(
      env, Napi::Function::New(env, NoOp), "musicLibraryCallback", 0, 1);

  auto callback = [=](Napi::Env env, Napi::Function js_cb, const char *status) {
    deferred.Resolve(Napi::String::New(env, status));
  };

  if (@available(macOS 10.16, *)) {
    std::string auth_status = MusicLibraryAuthStatus();

    if (auth_status == kNotDetermined) {
      [SKCloudServiceController
          requestAuthorization:^(SKCloudServiceAuthorizationStatus status) {
            tsfn.BlockingCall(StringFromMusicLibraryStatus(status).c_str(),
                              callback);
            tsfn.Release();
          }];
    } else {
      if (auth_status == kDenied)
        OpenPrefPane("Privacy_Media");

      tsfn.Release();
      deferred.Resolve(Napi::String::New(env, auth_status));
    }
  } else {
    tsfn.Release();
    deferred.Resolve(Napi::String::New(env, kAuthorized));
  }

  return deferred.Promise();
}

// Request Screen Capture Access.
void AskForScreenCaptureAccess(const Napi::CallbackInfo &info) {
  if (@available(macOS 11.0, *)) {
    if (CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess())
      return;

    bool should_force_prefs = info[0].As<Napi::Boolean>().Value();
    if (should_force_prefs && !HasOpenSystemPreferencesDialog()) {
      OpenPrefPane("Privacy_ScreenCapture");
    }
  } else {
    // Tries to create a capture stream. This is necessary to add the app back
    // to the list in sysprefs if the user previously denied.
    // https://stackoverflow.com/questions/56597221/detecting-screen-recording-settings-on-macos-catalina
    CGDisplayStreamRef stream = CGDisplayStreamCreate(
        CGMainDisplayID(), 1, 1, kCVPixelFormatType_32BGRA, NULL,
        ^(CGDisplayStreamFrameStatus status, uint64_t displayTime,
          IOSurfaceRef frameSurface, CGDisplayStreamUpdateRef updateRef){
        });

    if (stream) {
      CFRelease(stream);
    } else {
      if (!HasOpenSystemPreferencesDialog())
        OpenPrefPane("Privacy_ScreenCapture");
    }
  }
}

// Request Location access.
void AskForLocationAccess(const Napi::CallbackInfo &info) {
  std::string auth_status = LocationAuthStatus();

  if (auth_status == kNotDetermined) {
    CLLocationManager *location_manager = [[CLLocationManager alloc] init];
    const std::string access_level = info[0].As<Napi::String>().Utf8Value();
    if (access_level == "always") {
      [location_manager requestAlwaysAuthorization];
    } else if (access_level == "when-in-use") {
      [location_manager requestWhenInUseAuthorization];
    }
  } else if (auth_status == kDenied) {
    OpenPrefPane("Privacy_Location");
  }
}

// Request Accessibility Access.
void AskForAccessibilityAccess(const Napi::CallbackInfo &info) {
  NSDictionary *options = @{(id)kAXTrustedCheckOptionPrompt : @(NO)};
  bool trusted = AXIsProcessTrustedWithOptions((CFDictionaryRef)options);

  if (!trusted) {
    OpenPrefPane("Privacy_Accessibility");
  }
}

// Initializes all functions exposed to JS
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "askForAccessibilityAccess"),
              Napi::Function::New(env, AskForAccessibilityAccess));
  exports.Set(Napi::String::New(env, "askForAppleEventsAccess"),
              Napi::Function::New(env, AskForAppleEventsAccess));
  exports.Set(Napi::String::New(env, "askForCalendarAccess"),
              Napi::Function::New(env, AskForCalendarAccess));
  exports.Set(Napi::String::New(env, "askForCameraAccess"),
              Napi::Function::New(env, AskForCameraAccess));
  exports.Set(Napi::String::New(env, "askForContactsAccess"),
              Napi::Function::New(env, AskForContactsAccess));
  exports.Set(Napi::String::New(env, "askForFoldersAccess"),
              Napi::Function::New(env, AskForFoldersAccess));
  exports.Set(Napi::String::New(env, "askForFullDiskAccess"),
              Napi::Function::New(env, AskForFullDiskAccess));
  exports.Set(Napi::String::New(env, "askForInputMonitoringAccess"),
              Napi::Function::New(env, AskForInputMonitoringAccess));
  exports.Set(Napi::String::New(env, "askForLocationAccess"),
              Napi::Function::New(env, AskForLocationAccess));
  exports.Set(Napi::String::New(env, "askForMicrophoneAccess"),
              Napi::Function::New(env, AskForMicrophoneAccess));
  exports.Set(Napi::String::New(env, "askForMusicLibraryAccess"),
              Napi::Function::New(env, AskForMusicLibraryAccess));
  exports.Set(Napi::String::New(env, "askForPhotosAccess"),
              Napi::Function::New(env, AskForPhotosAccess));
  exports.Set(Napi::String::New(env, "askForRemindersAccess"),
              Napi::Function::New(env, AskForRemindersAccess));
  exports.Set(Napi::String::New(env, "askForScreenCaptureAccess"),
              Napi::Function::New(env, AskForScreenCaptureAccess));
  exports.Set(Napi::String::New(env, "askForSpeechRecognitionAccess"),
              Napi::Function::New(env, AskForSpeechRecognitionAccess));
  exports.Set(Napi::String::New(env, "getAuthStatus"),
              Napi::Function::New(env, GetAuthStatus));
  return exports;
}

NODE_API_MODULE(permissions, Init)
