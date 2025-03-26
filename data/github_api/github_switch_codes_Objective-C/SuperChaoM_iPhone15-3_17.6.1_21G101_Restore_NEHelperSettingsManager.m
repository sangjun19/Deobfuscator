// Repository: SuperChaoM/iPhone15-3_17.6.1_21G101_Restore
// File: usr/libexec/nehelper/NEHelperSettingsManager.m

@implementation NEHelperSettingsManager

- (void)handleMessage:(id)a3 
{
  id v3;
  const char *string;
  int64_t int64;
  _BOOL4 v6;
  NSObject *v7;
  const __SCPreferences *v8;
  const __SCPreferences *v9;
  const __CFString *v10;
  _BOOL4 v11;
  CFPropertyListRef *v12;
  NSObject *v13;
  NSObject *v14;
  NSObject *v15;
  NSObject *v16;
  NSObject *v17;
  int v18;
  NSObject *v19;
  void *Error;
  CFErrorRef v21;
  void *v22;
  void *v23;
  NSObject *v24;
  uint32_t v25;
  uint32_t v26;
  NSObject *v27;
  NSObject *v28;
  NSObject *v29;
  CFErrorRef v30;
  CFErrorRef v31;
  CFErrorRef v32;
  CFNumberRef v33;
  _BYTE buf[12];
  __int16 v35;
  CFErrorRef v36;

  v3 = objc_retain(a3);
  string = xpc_dictionary_get_string(v3, "setting-name");
  int64 = xpc_dictionary_get_int64(v3, "setting-type");
  if (string)
    v6 = strcmp(string, "CriticalDomains") == 0;
  else
    v6 = 0;
  ne_log_obj();
  v7 = (NSObject *)objc_claimAutoreleasedReturnValue();
  if (os_log_type_enabled(v7, OS_LOG_TYPE_DEFAULT))
  {
    *(_DWORD *)buf = 136315138;
    *(_QWORD *)&buf[4] = string;
    _os_log_impl((void *)&_mh_execute_header, v7, OS_LOG_TYPE_DEFAULT, "Handling a Settings message with setting name %s", buf, 0xCu);
  }
  objc_release(v7);
  v8 = SCPreferencesCreate(kCFAllocatorDefault, CFSTR("nehelper"), CFSTR("/Library/Preferences/com.apple.networkextension.control.plist"));
  if (v8)
  {
    v9 = v8;
    if (SCPreferencesLock(v8, 1u))
    {
      v10 = CFStringCreateWithCString(kCFAllocatorDefault, string, 0x600u);
      switch(int64)
      {
        case 1LL:
          v11 = xpc_dictionary_get_BOOL(v3, "setting-value");
          v12 = (CFPropertyListRef *)&kCFBooleanTrue;
          if (!v11)
            v12 = (CFPropertyListRef *)&kCFBooleanFalse;
          SCPreferencesSetValue(v9, v10, *v12);
          goto LABEL_39;
        case 2LL:
          *(_QWORD *)buf = xpc_dictionary_get_int64(v3, "setting-value");
          v33 = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, buf);
          SCPreferencesSetValue(v9, v10, v33);
          myCFRelease();
          goto LABEL_39;
        case 3LL:
          xpc_dictionary_get_value(v3, "setting-value");
          v15 = (NSObject *)objc_claimAutoreleasedReturnValue();
          v16 = v15;
          if (v15 && xpc_get_type(v15) == (xpc_type_t)&_xpc_type_array)
          {
            SCPreferencesGetValue(v9, v10);
            Error = (void *)objc_claimAutoreleasedReturnValue();
            v22 = (void *)_CFXPCCreateCFObjectFromXPCObject();
            sub_10000EA00((uint64_t)&OBJC_CLASS___NEHelperSettingsManager, Error, v22);
            v23 = (void *)objc_claimAutoreleasedReturnValue();
            if (v6)
            {
              ne_log_large_obj();
              v24 = (NSObject *)objc_claimAutoreleasedReturnValue();
              if (os_log_type_enabled(v24, OS_LOG_TYPE_DEFAULT))
              {
                *(_DWORD *)buf = 136315394;
                *(_QWORD *)&buf[4] = string;
                v35 = 2112;
                v36 = (CFErrorRef)v22;
                _os_log_impl((void *)&_mh_execute_header, v24, OS_LOG_TYPE_DEFAULT, "Saving new %s: %@", buf, 0x16u);
              }
              objc_release(v24);
            }
            SCPreferencesSetValue(v9, v10, v23);
            objc_release(v23);
            objc_release(v22);
            goto LABEL_37;
          }
          ne_log_obj();
          v17 = (NSObject *)objc_claimAutoreleasedReturnValue();
          if (os_log_type_enabled(v17, OS_LOG_TYPE_ERROR))
          {
            *(_WORD *)buf = 0;
            _os_log_error_impl((void *)&_mh_execute_header, v17, OS_LOG_TYPE_ERROR, "Setting value is not a valid array", buf, 2u);
          }
          objc_release(v17);
LABEL_29:
          objc_release(v16);
          goto LABEL_45;
        case 4LL:
          v18 = SCPreferencesRemoveValue(v9, v10);
          ne_log_obj();
          v19 = (NSObject *)objc_claimAutoreleasedReturnValue();
          v16 = v19;
          if (v18)
          {
            if (os_log_type_enabled(v19, OS_LOG_TYPE_DEFAULT))
            {
              *(_DWORD *)buf = 138412290;
              *(_QWORD *)&buf[4] = v10;
              _os_log_impl((void *)&_mh_execute_header, v16, OS_LOG_TYPE_DEFAULT, "Removed %@ setting", buf, 0xCu);
            }
          }
          else if (os_log_type_enabled(v19, OS_LOG_TYPE_ERROR))
          {
            Error = SCCopyLastError();
            *(_DWORD *)buf = 138412546;
            *(_QWORD *)&buf[4] = v10;
            v35 = 2112;
            v36 = (CFErrorRef)Error;
            _os_log_error_impl((void *)&_mh_execute_header, v16, OS_LOG_TYPE_ERROR, "Failed to remove %@ setting: %@", buf, 0x16u);
LABEL_37:
            objc_release(Error);
          }
          objc_release(v16);
LABEL_39:
          if (SCPreferencesCommitChanges(v9))
          {
            sub_1000021A4((uint64_t)&OBJC_CLASS___NEHelperServer, v3, 0LL, 0LL);
            if (v6)
            {
              v25 = notify_post("com.apple.neconfigurationchanged");
              if (v25)
              {
                v26 = v25;
                ne_log_obj();
                v27 = (NSObject *)objc_claimAutoreleasedReturnValue();
                if (os_log_type_enabled(v27, OS_LOG_TYPE_ERROR))
                {
                  *(_DWORD *)buf = 136315394;
                  *(_QWORD *)&buf[4] = "com.apple.neconfigurationchanged";
                  v35 = 1024;
                  LODWORD(v36) = v26;
                  _os_log_error_impl((void *)&_mh_execute_header, v27, OS_LOG_TYPE_ERROR, "Failed to post %s: %d", buf, 0x12u);
                }
                objc_release(v27);
              }
            }
          }
          else
          {
LABEL_45:
            ne_log_obj();
            v28 = (NSObject *)objc_claimAutoreleasedReturnValue();
            if (os_log_type_enabled(v28, OS_LOG_TYPE_ERROR))
            {
              v31 = SCCopyLastError();
              *(_DWORD *)buf = 136315394;
              *(_QWORD *)&buf[4] = string;
              v35 = 2112;
              v36 = v31;
              _os_log_error_impl((void *)&_mh_execute_header, v28, OS_LOG_TYPE_ERROR, "Failed to change %s: %@", buf, 0x16u);
              objc_release(v31);
            }
            objc_release(v28);
            sub_1000021A4((uint64_t)&OBJC_CLASS___NEHelperServer, v3, 22LL, 0LL);
          }
          myCFRelease();
          if (!SCPreferencesUnlock(v9))
          {
            ne_log_obj();
            v29 = (NSObject *)objc_claimAutoreleasedReturnValue();
            if (os_log_type_enabled(v29, OS_LOG_TYPE_ERROR))
            {
              v32 = SCCopyLastError();
              *(_DWORD *)buf = 136315394;
              *(_QWORD *)&buf[4] = "/Library/Preferences/com.apple.networkextension.control.plist";
              v35 = 2112;
              v36 = v32;
              _os_log_error_impl((void *)&_mh_execute_header, v29, OS_LOG_TYPE_ERROR, "Failed to unlock preferences for %s: %@", buf, 0x16u);
              objc_release(v32);
            }
            objc_release(v29);
          }
          break;
        default:
          ne_log_obj();
          v16 = (NSObject *)objc_claimAutoreleasedReturnValue();
          if (os_log_type_enabled(v16, OS_LOG_TYPE_ERROR))
          {
            *(_DWORD *)buf = 134217984;
            *(_QWORD *)&buf[4] = int64;
            _os_log_error_impl((void *)&_mh_execute_header, v16, OS_LOG_TYPE_ERROR, "Invalid setting type: %lld", buf, 0xCu);
          }
          goto LABEL_29;
      }
    }
    else
    {
      ne_log_obj();
      v14 = (NSObject *)objc_claimAutoreleasedReturnValue();
      if (os_log_type_enabled(v14, OS_LOG_TYPE_ERROR))
      {
        v30 = SCCopyLastError();
        *(_DWORD *)buf = 136315394;
        *(_QWORD *)&buf[4] = "/Library/Preferences/com.apple.networkextension.control.plist";
        v35 = 2112;
        v36 = v30;
        _os_log_error_impl((void *)&_mh_execute_header, v14, OS_LOG_TYPE_ERROR, "Failed to lock SCPreferences for %s: %@", buf, 0x16u);
        objc_release(v30);
      }
      objc_release(v14);
      sub_1000021A4((uint64_t)&OBJC_CLASS___NEHelperServer, v3, 22LL, 0LL);
    }
    CFRelease(v9);
  }
  else
  {
    ne_log_obj();
    v13 = (NSObject *)objc_claimAutoreleasedReturnValue();
    if (os_log_type_enabled(v13, OS_LOG_TYPE_ERROR))
    {
      v21 = SCCopyLastError();
      *(_DWORD *)buf = 136315394;
      *(_QWORD *)&buf[4] = "/Library/Preferences/com.apple.networkextension.control.plist";
      v35 = 2112;
      v36 = v21;
      _os_log_error_impl((void *)&_mh_execute_header, v13, OS_LOG_TYPE_ERROR, "Failed to create SCPreferences for %s: %@", buf, 0x16u);
      objc_release(v21);
    }
    objc_release(v13);
    sub_1000021A4((uint64_t)&OBJC_CLASS___NEHelperServer, v3, 22LL, 0LL);
  }
  objc_release(v3);
}

- (NEHelperSettingsManager)initWithFirstMessage:(id)a3 
{
  void *v4;
  NEHelperSettingsManager *v5;
  NSObject *v6;
  id v8;
  objc_super v9;
  uint8_t buf[4];
  id v11;
  __int16 v12;
  const char *v13;

  xpc_dictionary_get_remote_connection(a3);
  v4 = (void *)objc_claimAutoreleasedReturnValue();
  if (sub_10000DFD4((uint64_t)&OBJC_CLASS___NEHelperServer, v4))
  {
    v9.receiver = self;
    v9.super_class = (Class)&OBJC_CLASS___NEHelperSettingsManager;
    self = objc_retain(-[NEHelperSettingsManager init](&v9, "init"));
    v5 = self;
  }
  else
  {
    ne_log_obj();
    v6 = (NSObject *)objc_claimAutoreleasedReturnValue();
    if (os_log_type_enabled(v6, OS_LOG_TYPE_ERROR))
    {
      v8 = sub_10000E034((uint64_t)&OBJC_CLASS___NEHelperServer, v4);
      *(_DWORD *)buf = 138412546;
      v11 = v8;
      v12 = 2080;
      v13 = "com.apple.private.networkextension.configuration";
      _os_log_error_impl((void *)&_mh_execute_header, v6, OS_LOG_TYPE_ERROR, "Denying settings manager connection because %@ does not have the %s entitlement", buf, 0x16u);
      objc_release(v8);
    }
    objc_release(v6);
    v5 = 0LL;
  }
  objc_release(v4);
  objc_release(self);
  return v5;
}

@end
