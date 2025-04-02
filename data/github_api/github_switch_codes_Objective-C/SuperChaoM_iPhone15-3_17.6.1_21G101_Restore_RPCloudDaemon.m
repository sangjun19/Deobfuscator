// Repository: SuperChaoM/iPhone15-3_17.6.1_21G101_Restore
// File: usr/libexec/rapportd/RPCloudDaemon.m

@implementation RPCloudDaemon

+ (id)sharedCloudDaemon
{
  void *v2;
  uint64_t vars8;

  if (qword_10012D6A0 != -1)
    dispatch_once(&qword_10012D6A0, &stru_10010CC68);
  v2 = (void *)qword_10012D698;
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_retainAutoreleaseReturnValue(v2);
}

- (BOOL)idsHasWatch
{
  RPCloudDaemon *v3;
  void *v4;
  id v5;
  uint64_t v6;
  void *i;
  id v8;
  unsigned __int8 v9;
  BOOL v10;
  __int128 v12;
  __int128 v13;
  __int128 v14;
  __int128 v15;
  _BYTE v16[128];

  if (!-[RPCloudDaemon idsIsSignedIn](self, "idsIsSignedIn"))
    return 0;
  v3 = objc_retain(self);
  objc_sync_enter(v3);
  v12 = 0u;
  v13 = 0u;
  v14 = 0u;
  v15 = 0u;
  -[RPCloudDaemon idsDeviceArray](v3, "idsDeviceArray", 0LL);
  v4 = (void *)objc_claimAutoreleasedReturnValue();
  v5 = objc_msgSend(v4, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
  if (v5)
  {
    v6 = *(_QWORD *)v13;
    while (2)
    {
      for (i = 0LL; i != v5; i = (char *)i + 1)
      {
        if (*(_QWORD *)v13 != v6)
          objc_enumerationMutation(v4);
        objc_msgSend(*(id *)(*((_QWORD *)&v12 + 1) + 8LL * (_QWORD)i), "modelIdentifier");
        v8 = objc_retain((id)objc_claimAutoreleasedReturnValue());
        v9 = (unsigned __int8)objc_msgSend(v8, "hasPrefix:", CFSTR("Watch"));
        objc_release(v8);
        objc_release(v8);
        if ((v9 & 1) != 0)
        {
          v10 = 1;
          goto LABEL_13;
        }
      }
      v5 = objc_msgSend(v4, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
      if (v5)
        continue;
      break;
    }
  }
  v10 = 0;
LABEL_13:
  objc_release(v4);
  objc_sync_exit(v3);
  objc_release(v3);
  return v10;
}

- (BOOL)idsIsSignedIn
{
  RPCloudDaemon *v2;
  int prefIsSignedInForce;
  BOOL v4;
  IDSService *nearbyIDSService;
  void *v6;
  id v7;
  uint64_t v8;
  void *i;
  __int128 v11;
  __int128 v12;
  __int128 v13;
  __int128 v14;
  _BYTE v15[128];

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  prefIsSignedInForce = v2->_prefIsSignedInForce;
  if (prefIsSignedInForce < 0)
  {
    if (v2->_idsIsSignedInCache < 0)
    {
      nearbyIDSService = v2->_nearbyIDSService;
      if (nearbyIDSService)
      {
        v2->_idsIsSignedInCache = 0;
        v11 = 0u;
        v12 = 0u;
        v13 = 0u;
        v14 = 0u;
        -[IDSService accounts](nearbyIDSService, "accounts", 0LL);
        v6 = (void *)objc_claimAutoreleasedReturnValue();
        v7 = objc_msgSend(v6, "countByEnumeratingWithState:objects:count:", &v11, v15, 16LL);
        if (v7)
        {
          v8 = *(_QWORD *)v12;
          while (2)
          {
            for (i = 0LL; i != v7; i = (char *)i + 1)
            {
              if (*(_QWORD *)v12 != v8)
                objc_enumerationMutation(v6);
              if (((unsigned int)objc_msgSend(*(id *)(*((_QWORD *)&v11 + 1) + 8LL * (_QWORD)i), "isActive") & 1) != 0)
              {
                v2->_idsIsSignedInCache = 1;
                goto LABEL_15;
              }
            }
            v7 = objc_msgSend(v6, "countByEnumeratingWithState:objects:count:", &v11, v15, 16LL);
            if (v7)
              continue;
            break;
          }
        }
LABEL_15:
        objc_release(v6);
      }
    }
    v4 = v2->_idsIsSignedInCache > 0;
  }
  else
  {
    v4 = prefIsSignedInForce != 0;
  }
  objc_sync_exit(v2);
  objc_release(v2);
  return v4;
}

- (NSString)idsDeviceIDSelf
{
  RPCloudDaemon *v2;
  NSString *v3;
  uint64_t vars8;

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  v3 = objc_retain(v2->_idsDeviceIDSelf);
  if (!v3)
  {
    v3 = (NSString *)IDSCopyLocalDeviceUniqueID();
    if (v3)
    {
      objc_storeStrong((id *)&v2->_idsDeviceIDSelf, v3);
    }
    else
    {
      if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
        LogPrintF();
      v3 = 0LL;
    }
  }
  objc_sync_exit(v2);
  objc_release(v2);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_autoreleaseReturnValue(v3);
}

- (id)idsCorrelationIdentifier
{
  NSString *idsCorrelationIdentifier;
  void *v4;
  void *v5;
  void *v6;
  void *v7;
  OS_dispatch_queue *dispatchQueue;
  _QWORD v10[6];
  uint64_t vars8;

  idsCorrelationIdentifier = self->_idsCorrelationIdentifier;
  if (!idsCorrelationIdentifier)
  {
    if (self->_idQueryInProgress)
    {
      idsCorrelationIdentifier = 0LL;
    }
    else
    {
      -[CUSystemMonitor primaryAppleID](self->_systemMonitor, "primaryAppleID");
      v4 = (void *)objc_claimAutoreleasedReturnValue();
      objc_msgSend(v4, "_bestGuessURI");
      v5 = (void *)objc_claimAutoreleasedReturnValue();
      if (v5)
      {
        +[NSArray arrayWithObject:](&OBJC_CLASS___NSArray, "arrayWithObject:", v5);
        v6 = (void *)objc_claimAutoreleasedReturnValue();
        +[IDSIDQueryController sharedInstance](&OBJC_CLASS___IDSIDQueryController, "sharedInstance");
        v7 = (void *)objc_claimAutoreleasedReturnValue();
        dispatchQueue = self->_dispatchQueue;
        v10[0] = _NSConcreteStackBlock;
        v10[1] = 3221225472LL;
        v10[2] = sub_100011578;
        v10[3] = &unk_10010CCB8;
        v10[4] = self;
        v10[5] = v4;
        objc_msgSend(v7, "currentRemoteDevicesForDestinations:service:listenerID:queue:completionBlock:", v6, CFSTR("com.apple.private.alloy.nearby"), CFSTR("com.apple.private.alloy.nearby"), dispatchQueue, v10);
        self->_idQueryInProgress = 1;
        objc_release(v7);
        objc_release(v6);
      }
      objc_release(v5);
      objc_release(v4);
      idsCorrelationIdentifier = self->_idsCorrelationIdentifier;
    }
  }
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_retainAutoreleaseReturnValue(idsCorrelationIdentifier);
}

- (NSDictionary)idsFamilyEndpointMap
{
  NSDictionary *idsFamilyEndpointMap;
  uint64_t vars8;

  dispatch_assert_queue_V2((dispatch_queue_t)self->_dispatchQueue);
  idsFamilyEndpointMap = self->_idsFamilyEndpointMap;
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_retainAutoreleaseReturnValue(idsFamilyEndpointMap);
}

- (RPCloudDaemon)init
{
  RPCloudDaemon *v2;
  RPCloudDaemon *v3;
  RPCloudDaemon *v4;
  objc_super v6;

  v6.receiver = self;
  v6.super_class = (Class)&OBJC_CLASS___RPCloudDaemon;
  v2 = -[RPCloudDaemon init](&v6, "init");
  v3 = v2;
  if (v2)
  {
    objc_storeStrong((id *)&v2->_dispatchQueue, &_dispatch_main_q);
    v3->_prefIsSignedInForce = -1;
    *(_QWORD *)&v3->_idsHandheldCountCache = -1LL;
    *(_QWORD *)&v3->_idsHasHomePodCache = -1LL;
    *(_QWORD *)&v3->_idsHasMacCache = -1LL;
    v3->_idsIsSignedInCache = -1;
    *(_QWORD *)&v3->_prefHasAppleTVForce = -1LL;
    *(_QWORD *)&v3->_prefHasiPadForce = -1LL;
    v3->_prefHasRealityDeviceForce = -1;
    v4 = objc_retain(v3);
  }
  objc_release(v3);
  return v3;
}

- (id)descriptionWithLevel:(int)a3 
{
  __CFString *v3;
  void *v4;
  __CFString *v5;
  uint64_t v6;
  void *v7;
  void *v8;
  void *v9;
  void *v10;
  void *v11;
  void *v12;
  void *v13;
  void *v14;
  void *v15;
  id v16;
  id v17;
  id v18;
  id obj;
  uint64_t v22;
  id v24;
  __int128 v25;
  __int128 v26;
  __int128 v27;
  __int128 v28;
  _BYTE v29[128];
  uint64_t vars8;

  v25 = 0u;
  v26 = 0u;
  v27 = 0u;
  v28 = 0u;
  -[NSDictionary allKeys](self->_idsFamilyEndpointMap, "allKeys");
  obj = (id)objc_claimAutoreleasedReturnValue();
  v24 = objc_msgSend(obj, "countByEnumeratingWithState:objects:count:", &v25, v29, 16LL);
  if (v24)
  {
    v22 = *(_QWORD *)v26;
    v3 = &stru_1001107E8;
    do
    {
      v4 = 0LL;
      v5 = v3;
      do
      {
        if (*(_QWORD *)v26 != v22)
          objc_enumerationMutation(obj);
        v6 = *(_QWORD *)(*((_QWORD *)&v25 + 1) + 8LL * (_QWORD)v4);
        -[NSDictionary objectForKeyedSubscript:](self->_idsFamilyEndpointMap, "objectForKeyedSubscript:", v6);
        v7 = (void *)objc_claimAutoreleasedReturnValue();
        objc_msgSend(v7, "familyEndpointData");
        v8 = (void *)objc_claimAutoreleasedReturnValue();
        objc_msgSend(v8, "deviceName");
        v9 = (void *)objc_claimAutoreleasedReturnValue();
        -[NSDictionary objectForKeyedSubscript:](self->_idsFamilyEndpointMap, "objectForKeyedSubscript:", v6);
        v10 = (void *)objc_claimAutoreleasedReturnValue();
        objc_msgSend(v10, "familyEndpointData");
        v11 = (void *)objc_claimAutoreleasedReturnValue();
        objc_msgSend(v11, "productVersion");
        v12 = (void *)objc_claimAutoreleasedReturnValue();
        +[NSString stringWithFormat:](&OBJC_CLASS___NSString, "stringWithFormat:", CFSTR("    IDSID: %@, name: %@, productVersion: %@\n"), v6, v9, v12);
        v13 = (void *)objc_claimAutoreleasedReturnValue();
        objc_release(v12);
        objc_release(v11);
        objc_release(v10);
        objc_release(v9);
        objc_release(v8);
        objc_release(v7);
        -[__CFString stringByAppendingString:](v5, "stringByAppendingString:", v13);
        v3 = (__CFString *)objc_claimAutoreleasedReturnValue();
        objc_release(v5);
        objc_release(v13);
        v4 = (char *)v4 + 1;
        v5 = v3;
      }
      while (v24 != v4);
      v24 = objc_msgSend(obj, "countByEnumeratingWithState:objects:count:", &v25, v29, 16LL);
    }
    while (v24);
  }
  else
  {
    v3 = &stru_1001107E8;
  }
  objc_release(obj);
  -[RPCloudDaemon idsDeviceArray](self, "idsDeviceArray");
  v14 = (void *)objc_claimAutoreleasedReturnValue();
  objc_msgSend(v14, "count");
  -[RPCloudDaemon idsIsSignedIn](self, "idsIsSignedIn");
  -[RPCloudDaemon idsHasAppleTV](self, "idsHasAppleTV");
  -[RPCloudDaemon idsHasHomePod](self, "idsHasHomePod");
  -[RPCloudDaemon idsHasiPad](self, "idsHasiPad");
  -[RPCloudDaemon idsHasMac](self, "idsHasMac");
  -[RPCloudDaemon idsHasWatch](self, "idsHasWatch");
  -[RPCloudDaemon idsHasRealityDevice](self, "idsHasRealityDevice");
  -[RPCloudDaemon idsHandheldCount](self, "idsHandheldCount");
  -[RPCloudDaemon idsFamilyEndpointMap](self, "idsFamilyEndpointMap");
  v15 = (void *)objc_claimAutoreleasedReturnValue();
  objc_msgSend(v15, "count");
  NSAppendPrintF();
  v16 = objc_retain(0LL);
  objc_release(v15);
  objc_release(v14);
  if (a3 <= 20)
  {
    NSAppendPrintF();
    v17 = objc_retain(v16);
    objc_release(v16);
    v16 = v17;
  }
  v18 = objc_retain(v16);
  objc_release(v3);
  objc_release(v18);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_autoreleaseReturnValue(v18);
}

- (void)activate
{
  NSObject *dispatchQueue;
  _QWORD block[5];

  dispatchQueue = (NSObject *)self->_dispatchQueue;
  block[0] = _NSConcreteStackBlock;
  block[1] = 3221225472LL;
  block[2] = sub_100010734;
  block[3] = &unk_10010CC90;
  block[4] = self;
  dispatch_async(dispatchQueue, block);
}

- (void)invalidate
{
  NSObject *dispatchQueue;
  _QWORD block[5];

  dispatchQueue = (NSObject *)self->_dispatchQueue;
  block[0] = _NSConcreteStackBlock;
  block[1] = 3221225472LL;
  block[2] = sub_100010964;
  block[3] = &unk_10010CC90;
  block[4] = self;
  dispatch_async(dispatchQueue, block);
}

- (void)_invalidate
{
  CUSystemMonitor *systemMonitor;
  uint64_t vars8;

  if (!self->_invalidateCalled)
  {
    self->_invalidateCalled = 1;
    if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      LogPrintF();
    -[CUSystemMonitor invalidate](self->_systemMonitor, "invalidate");
    systemMonitor = self->_systemMonitor;
    self->_systemMonitor = 0LL;
    objc_release(systemMonitor);
    -[RPCloudDaemon _idsEnsureStopped](self, "_idsEnsureStopped");
    if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
      __break(0xC471u);
    -[RPCloudDaemon _invalidated](self, "_invalidated");
  }
}

- (void)_invalidated
{
  uint64_t vars8;

  if (self->_invalidateCalled && !self->_invalidateDone)
  {
    self->_invalidateDone = 1;
    if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    {
      if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
        __break(0xC471u);
      LogPrintF();
    }
  }
}

- (void)daemonInfoChanged:(unint64_t)a3 
{
  uint64_t vars8;

  if ((a3 & 0x400) != 0)
  {
    if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      LogPrintF();
    if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
      __break(0xC471u);
    -[RPCloudDaemon idsFamilyEndpointsUpdateWithForce:](self, "idsFamilyEndpointsUpdateWithForce:", 1LL);
  }
}

- (BOOL)diagnosticCommand:(id)a3 params:(id)a4 
{
  dispatch_assert_queue_V2((dispatch_queue_t)self->_dispatchQueue);
  return 0;
}

- (void)prefsChanged
{
  _BOOL8 v3;
  const char *prefHasAppleTVForce;
  _BOOL8 v5;
  const char *prefHasHomePodForce;
  _BOOL8 v7;
  const char *prefHasiPadForce;
  _BOOL8 v9;
  const char *prefHasMacForce;
  _BOOL8 v11;
  const char *prefHasRealityDeviceForce;
  _BOOL4 v13;
  const char *v14;
  const char *v15;
  _BOOL8 v16;
  const char *prefIsSignedInForce;
  const char *v18;
  const char *v19;

  v3 = CFPrefs_GetInt64() != 0;
  prefHasAppleTVForce = (const char *)(unsigned int)self->_prefHasAppleTVForce;
  if (v3 != (_DWORD)prefHasAppleTVForce)
  {
    if (dword_10012BA38 <= 40)
    {
      if (dword_10012BA38 == -1)
      {
        if (!(unsigned int)_LogCategory_Initialize())
          goto LABEL_6;
        prefHasAppleTVForce = (const char *)(unsigned int)self->_prefHasAppleTVForce;
      }
      v18 = prefHasAppleTVForce;
      v19 = (const char *)v3;
      LogPrintF();
    }
LABEL_6:
    self->_prefHasAppleTVForce = v3;
  }
  v5 = CFPrefs_GetInt64() != 0;
  prefHasHomePodForce = (const char *)(unsigned int)self->_prefHasHomePodForce;
  if (v5 == (_DWORD)prefHasHomePodForce)
    goto LABEL_13;
  if (dword_10012BA38 <= 40)
  {
    if (dword_10012BA38 == -1)
    {
      if (!(unsigned int)_LogCategory_Initialize())
        goto LABEL_12;
      prefHasHomePodForce = (const char *)(unsigned int)self->_prefHasHomePodForce;
    }
    v18 = prefHasHomePodForce;
    v19 = (const char *)v5;
    LogPrintF();
  }
LABEL_12:
  self->_prefHasHomePodForce = v5;
LABEL_13:
  v7 = CFPrefs_GetInt64() != 0;
  prefHasiPadForce = (const char *)(unsigned int)self->_prefHasiPadForce;
  if (v7 == (_DWORD)prefHasiPadForce)
    goto LABEL_19;
  if (dword_10012BA38 <= 40)
  {
    if (dword_10012BA38 == -1)
    {
      if (!(unsigned int)_LogCategory_Initialize())
        goto LABEL_18;
      prefHasiPadForce = (const char *)(unsigned int)self->_prefHasiPadForce;
    }
    v18 = prefHasiPadForce;
    v19 = (const char *)v7;
    LogPrintF();
  }
LABEL_18:
  self->_prefHasiPadForce = v7;
LABEL_19:
  v9 = CFPrefs_GetInt64() != 0;
  prefHasMacForce = (const char *)(unsigned int)self->_prefHasMacForce;
  if (v9 == (_DWORD)prefHasMacForce)
    goto LABEL_25;
  if (dword_10012BA38 <= 40)
  {
    if (dword_10012BA38 == -1)
    {
      if (!(unsigned int)_LogCategory_Initialize())
        goto LABEL_24;
      prefHasMacForce = (const char *)(unsigned int)self->_prefHasMacForce;
    }
    v18 = prefHasMacForce;
    v19 = (const char *)v9;
    LogPrintF();
  }
LABEL_24:
  self->_prefHasMacForce = v9;
LABEL_25:
  v11 = CFPrefs_GetInt64() != 0;
  prefHasRealityDeviceForce = (const char *)(unsigned int)self->_prefHasRealityDeviceForce;
  if (v11 == (_DWORD)prefHasRealityDeviceForce)
    goto LABEL_31;
  if (dword_10012BA38 <= 40)
  {
    if (dword_10012BA38 == -1)
    {
      if (!(unsigned int)_LogCategory_Initialize())
        goto LABEL_30;
      prefHasRealityDeviceForce = (const char *)(unsigned int)self->_prefHasRealityDeviceForce;
    }
    v18 = prefHasRealityDeviceForce;
    v19 = (const char *)v11;
    LogPrintF();
  }
LABEL_30:
  self->_prefHasRealityDeviceForce = v11;
LABEL_31:
  v13 = CFPrefs_GetInt64() != 0;
  if (self->_prefIDSEnabled != v13)
  {
    if (dword_10012BA38 <= 40 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    {
      v14 = "yes";
      if (v13)
        v15 = "no";
      else
        v15 = "yes";
      if (!v13)
        v14 = "no";
      v18 = v15;
      v19 = v14;
      LogPrintF();
    }
    self->_prefIDSEnabled = v13;
  }
  v16 = CFPrefs_GetInt64() != 0;
  prefIsSignedInForce = (const char *)(unsigned int)self->_prefIsSignedInForce;
  if (v16 != (_DWORD)prefIsSignedInForce)
  {
    if (dword_10012BA38 <= 40)
    {
      if (dword_10012BA38 == -1)
      {
        if (!(unsigned int)_LogCategory_Initialize())
          goto LABEL_47;
        prefIsSignedInForce = (const char *)(unsigned int)self->_prefIsSignedInForce;
      }
      v18 = prefIsSignedInForce;
      v19 = (const char *)v16;
      LogPrintF();
    }
LABEL_47:
    self->_prefIsSignedInForce = v16;
  }
  -[RPCloudDaemon _update](self, "_update", v18, v19);
}

- (BOOL)addXPCMatchingToken:(unint64_t)a3 event:(id)a4 handler:(id)a5 
{
  return 0;
}

- (BOOL)removeXPCMatchingToken:(unint64_t)a3 
{
  return 0;
}

- (void)_update
{
  if (self->_prefIDSEnabled)
    -[RPCloudDaemon _idsEnsureStarted](self, "_idsEnsureStarted");
  else
    -[RPCloudDaemon _idsEnsureStopped](self, "_idsEnsureStopped");
}

- (void)_primaryAppleIDChanged
{
  NSString *idsCorrelationIdentifier;

  idsCorrelationIdentifier = self->_idsCorrelationIdentifier;
  self->_idsCorrelationIdentifier = 0LL;
  objc_release(idsCorrelationIdentifier);
}

- (void)_idsEnsureStarted
{
  IDSService *v3;
  IDSService *nearbyIDSService;
  IDSService *v5;
  OS_dispatch_queue *dispatchQueue;
  uint64_t vars8;

  if (!self->_nearbyIDSService)
  {
    if (dword_10012BA38 <= 40 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      LogPrintF();
    v3 = (IDSService *)objc_msgSend(objc_alloc((Class)&OBJC_CLASS___IDSService), "initWithService:", CFSTR("com.apple.private.alloy.nearby"));
    nearbyIDSService = self->_nearbyIDSService;
    self->_nearbyIDSService = v3;
    objc_release(nearbyIDSService);
    dispatchQueue = self->_dispatchQueue;
    v5 = self->_nearbyIDSService;
    if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
      __break(0xC471u);
    -[IDSService addDelegate:queue:](v5, "addDelegate:queue:", self, dispatchQueue);
  }
}

- (void)_idsEnsureStopped
{
  IDSService *nearbyIDSService;
  NSString *idsCorrelationIdentifier;
  uint64_t vars8;

  if (self->_nearbyIDSService)
  {
    if (dword_10012BA38 <= 40 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      LogPrintF();
    -[IDSService removeDelegate:](self->_nearbyIDSService, "removeDelegate:", self);
    nearbyIDSService = self->_nearbyIDSService;
    self->_nearbyIDSService = 0LL;
    objc_release(nearbyIDSService);
  }
  idsCorrelationIdentifier = self->_idsCorrelationIdentifier;
  self->_idsCorrelationIdentifier = 0LL;
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(idsCorrelationIdentifier);
}

- (id)_idsAccountWithURI:(id)a3 senderID:(id *)a4 
{
  id v6;
  void *v7;
  void *v8;
  id v9;
  id v10;
  uint64_t v11;
  void *i;
  void *v13;
  void *v14;
  id v15;
  id v16;
  uint64_t v17;
  void *j;
  void *v19;
  id v20;
  id v21;
  uint64_t v22;
  void *v23;
  void *v24;
  void *v25;
  id v26;
  id v27;
  uint64_t v28;
  void *k;
  uint64_t v30;
  void *v31;
  id v32;
  uint64_t v33;
  id v34;
  uint64_t v36;
  id v37;
  uint64_t v38;
  id v39;
  uint64_t v40;
  void *v41;
  RPCloudDaemon *v42;
  RPCloudDaemon *v43;
  id *v44;
  id *v45;
  id v46;
  __int128 v47;
  __int128 v48;
  __int128 v49;
  __int128 v50;
  __int128 v51;
  __int128 v52;
  __int128 v53;
  __int128 v54;
  __int128 v55;
  __int128 v56;
  __int128 v57;
  __int128 v58;
  __int128 v59;
  __int128 v60;
  __int128 v61;
  __int128 v62;
  _BYTE v63[128];
  _BYTE v64[128];
  _BYTE v65[128];
  _BYTE v66[128];
  uint64_t vars8;

  v6 = objc_retain(a3);
  v7 = (void *)IDSCopyRawAddressForDestination();
  v59 = 0u;
  v60 = 0u;
  v61 = 0u;
  v62 = 0u;
  -[IDSService accounts](self->_nearbyIDSService, "accounts");
  v8 = (void *)objc_claimAutoreleasedReturnValue();
  v9 = objc_msgSend(v8, "countByEnumeratingWithState:objects:count:", &v59, v66, 16LL);
  if (v9)
  {
    v10 = v9;
    v11 = *(_QWORD *)v60;
    v44 = a4;
    v46 = v6;
    v40 = *(_QWORD *)v60;
    v42 = self;
    do
    {
      for (i = 0LL; i != v10; i = (char *)i + 1)
      {
        if (*(_QWORD *)v60 != v11)
          objc_enumerationMutation(v8);
        v13 = *(void **)(*((_QWORD *)&v59 + 1) + 8LL * (_QWORD)i);
        if ((unsigned int)objc_msgSend(v13, "canSend"))
        {
          v57 = 0u;
          v58 = 0u;
          v55 = 0u;
          v56 = 0u;
          objc_msgSend(v13, "aliasStrings");
          v14 = (void *)objc_claimAutoreleasedReturnValue();
          v15 = objc_msgSend(v14, "countByEnumeratingWithState:objects:count:", &v55, v65, 16LL);
          if (v15)
          {
            v16 = v15;
            v17 = *(_QWORD *)v56;
            while (2)
            {
              for (j = 0LL; j != v16; j = (char *)j + 1)
              {
                if (*(_QWORD *)v56 != v17)
                  objc_enumerationMutation(v14);
                v19 = *(void **)(*((_QWORD *)&v55 + 1) + 8LL * (_QWORD)j);
                if ((unsigned int)objc_msgSend(v19, "isEqual:", v7))
                {
                  if (v44)
                  {
                    -[RPCloudDaemon _idsURIWithID:](v42, "_idsURIWithID:", v19);
                    v33 = objc_claimAutoreleasedReturnValue();
                    v34 = *v44;
                    *v44 = (id)v33;
                    objc_release(v34);
                  }
                  v32 = objc_retain(v13);
                  goto LABEL_40;
                }
              }
              v16 = objc_msgSend(v14, "countByEnumeratingWithState:objects:count:", &v55, v65, 16LL);
              if (v16)
                continue;
              break;
            }
          }
          objc_release(v14);
          v11 = v40;
        }
      }
      v10 = objc_msgSend(v8, "countByEnumeratingWithState:objects:count:", &v59, v66, 16LL);
      a4 = v44;
      v6 = v46;
      self = v42;
    }
    while (v10);
  }
  objc_release(v8);
  if (objc_msgSend(v6, "_FZIDType") == (id)-1LL)
  {
    v32 = 0LL;
  }
  else
  {
    IMCanonicalizeFormattedString();
    v8 = (void *)objc_claimAutoreleasedReturnValue();
    v51 = 0u;
    v52 = 0u;
    v53 = 0u;
    v54 = 0u;
    -[IDSService accounts](self->_nearbyIDSService, "accounts");
    v14 = (void *)objc_claimAutoreleasedReturnValue();
    v20 = objc_msgSend(v14, "countByEnumeratingWithState:objects:count:", &v51, v64, 16LL);
    if (v20)
    {
      v21 = v20;
      v43 = self;
      v45 = a4;
      v46 = v6;
      v22 = *(_QWORD *)v52;
      v38 = *(_QWORD *)v52;
      do
      {
        v23 = 0LL;
        v39 = v21;
        do
        {
          if (*(_QWORD *)v52 != v22)
            objc_enumerationMutation(v14);
          v24 = *(void **)(*((_QWORD *)&v51 + 1) + 8LL * (_QWORD)v23);
          if ((unsigned int)objc_msgSend(v24, "canSend", v38))
          {
            v49 = 0u;
            v50 = 0u;
            v47 = 0u;
            v48 = 0u;
            v41 = v24;
            objc_msgSend(v24, "aliasStrings");
            v25 = (void *)objc_claimAutoreleasedReturnValue();
            v26 = objc_msgSend(v25, "countByEnumeratingWithState:objects:count:", &v47, v63, 16LL);
            if (v26)
            {
              v27 = v26;
              v28 = *(_QWORD *)v48;
              while (2)
              {
                for (k = 0LL; k != v27; k = (char *)k + 1)
                {
                  if (*(_QWORD *)v48 != v28)
                    objc_enumerationMutation(v25);
                  v30 = *(_QWORD *)(*((_QWORD *)&v47 + 1) + 8LL * (_QWORD)k);
                  IMCanonicalizeFormattedString();
                  v31 = (void *)objc_claimAutoreleasedReturnValue();
                  if ((unsigned int)objc_msgSend(v31, "isEqual:", v8))
                  {
                    if (v45)
                    {
                      -[RPCloudDaemon _idsURIWithID:](v43, "_idsURIWithID:", v30);
                      v36 = objc_claimAutoreleasedReturnValue();
                      v37 = *v45;
                      *v45 = (id)v36;
                      objc_release(v37);
                    }
                    v32 = objc_retain(v41);
                    objc_release(v31);
                    objc_release(v25);
                    goto LABEL_40;
                  }
                  objc_release(v31);
                }
                v27 = objc_msgSend(v25, "countByEnumeratingWithState:objects:count:", &v47, v63, 16LL);
                if (v27)
                  continue;
                break;
              }
            }
            objc_release(v25);
            v22 = v38;
            v21 = v39;
          }
          v23 = (char *)v23 + 1;
        }
        while (v23 != v21);
        v21 = objc_msgSend(v14, "countByEnumeratingWithState:objects:count:", &v51, v64, 16LL);
      }
      while (v21);
      v32 = 0LL;
LABEL_40:
      v6 = v46;
    }
    else
    {
      v32 = 0LL;
    }
    objc_release(v14);
    objc_release(v8);
  }
  objc_release(v7);
  objc_release(v6);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_autoreleaseReturnValue(v32);
}

- (NSArray)idsDeviceArray
{
  RPCloudDaemon *v2;
  NSArray *idsDeviceArray;
  uint64_t v4;
  NSArray *v5;
  NSArray *v6;
  uint64_t vars8;

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  idsDeviceArray = v2->_idsDeviceArray;
  if (!idsDeviceArray)
  {
    if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      LogPrintF();
    -[IDSService devices](v2->_nearbyIDSService, "devices");
    v4 = objc_claimAutoreleasedReturnValue();
    v5 = v2->_idsDeviceArray;
    v2->_idsDeviceArray = (NSArray *)v4;
    objc_release(v5);
    idsDeviceArray = v2->_idsDeviceArray;
  }
  v6 = objc_retain(idsDeviceArray);
  objc_sync_exit(v2);
  objc_release(v2);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_autoreleaseReturnValue(v6);
}

- (id)idsDeviceForBluetoothUUID:(id)a3 
{
  id v4;
  RPCloudDaemon *v5;
  void *v6;
  id v7;
  uint64_t v8;
  void *i;
  void *v10;
  void *v11;
  unsigned __int8 v12;
  __int128 v14;
  __int128 v15;
  __int128 v16;
  __int128 v17;
  _BYTE v18[128];
  uint64_t vars8;

  v4 = objc_retain(a3);
  v5 = objc_retain(self);
  objc_sync_enter(v5);
  v14 = 0u;
  v15 = 0u;
  v16 = 0u;
  v17 = 0u;
  -[RPCloudDaemon idsDeviceArray](v5, "idsDeviceArray", 0LL);
  v6 = (void *)objc_claimAutoreleasedReturnValue();
  v7 = objc_msgSend(v6, "countByEnumeratingWithState:objects:count:", &v14, v18, 16LL);
  if (v7)
  {
    v8 = *(_QWORD *)v15;
    while (2)
    {
      for (i = 0LL; i != v7; i = (char *)i + 1)
      {
        if (*(_QWORD *)v15 != v8)
          objc_enumerationMutation(v6);
        v10 = *(void **)(*((_QWORD *)&v14 + 1) + 8LL * (_QWORD)i);
        objc_msgSend(v10, "nsuuid");
        v11 = (void *)objc_claimAutoreleasedReturnValue();
        v12 = (unsigned __int8)objc_msgSend(v11, "isEqual:", v4);
        objc_release(v11);
        if ((v12 & 1) != 0)
        {
          v7 = objc_retain(v10);
          goto LABEL_11;
        }
      }
      v7 = objc_msgSend(v6, "countByEnumeratingWithState:objects:count:", &v14, v18, 16LL);
      if (v7)
        continue;
      break;
    }
  }
LABEL_11:
  objc_release(v6);
  objc_sync_exit(v5);
  objc_release(v5);
  objc_release(v4);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_autoreleaseReturnValue(v7);
}

- (NSDictionary)idsDeviceMap
{
  RPCloudDaemon *v2;
  NSDictionary *idsDeviceMap;
  void *v4;
  id v5;
  id v6;
  id v7;
  uint64_t v8;
  void *i;
  void *v10;
  void *v11;
  NSDictionary *v12;
  NSDictionary *v13;
  NSDictionary *v14;
  __int128 v16;
  __int128 v17;
  __int128 v18;
  __int128 v19;
  _BYTE v20[128];
  uint64_t vars8;

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  idsDeviceMap = v2->_idsDeviceMap;
  if (idsDeviceMap)
  {
LABEL_13:
    v14 = objc_retain(idsDeviceMap);
    goto LABEL_14;
  }
  -[RPCloudDaemon idsDeviceArray](v2, "idsDeviceArray");
  v4 = (void *)objc_claimAutoreleasedReturnValue();
  if (v4)
  {
    v5 = objc_alloc_init((Class)&OBJC_CLASS___NSMutableDictionary);
    v18 = 0u;
    v19 = 0u;
    v16 = 0u;
    v17 = 0u;
    v6 = objc_retain(v4);
    v7 = objc_msgSend(v6, "countByEnumeratingWithState:objects:count:", &v16, v20, 16LL);
    if (v7)
    {
      v8 = *(_QWORD *)v17;
      do
      {
        for (i = 0LL; i != v7; i = (char *)i + 1)
        {
          if (*(_QWORD *)v17 != v8)
            objc_enumerationMutation(v6);
          v10 = *(void **)(*((_QWORD *)&v16 + 1) + 8LL * (_QWORD)i);
          objc_msgSend(v10, "uniqueIDOverride", (_QWORD)v16);
          v11 = (void *)objc_claimAutoreleasedReturnValue();
          if (v11)
            objc_msgSend(v5, "setObject:forKeyedSubscript:", v10, v11);
          objc_release(v11);
        }
        v7 = objc_msgSend(v6, "countByEnumeratingWithState:objects:count:", &v16, v20, 16LL);
      }
      while (v7);
    }
    objc_release(v6);
    v12 = (NSDictionary *)objc_msgSend(v5, "copy");
    v13 = v2->_idsDeviceMap;
    v2->_idsDeviceMap = v12;
    objc_release(v13);
    objc_release(v5);
    objc_release(v6);
    idsDeviceMap = v2->_idsDeviceMap;
    goto LABEL_13;
  }
  v14 = 0LL;
LABEL_14:
  objc_sync_exit(v2);
  objc_release(v2);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_autoreleaseReturnValue(v14);
}

- (void)idsFamilyEndpointQueryWithCompletion:(id)a3 
{
  void *v4;
  void *v5;
  void *v6;
  void *v7;
  id v8;
  uint64_t vars8;

  v8 = objc_retain(a3);
  +[RPPeopleDaemon sharedPeopleDaemon](&OBJC_CLASS___RPPeopleDaemon, "sharedPeopleDaemon");
  v4 = (void *)objc_claimAutoreleasedReturnValue();
  objc_msgSend(v4, "getFamilyURIs");
  v5 = (void *)objc_claimAutoreleasedReturnValue();
  objc_release(v4);
  +[IDSIDQueryController sharedInstance](&OBJC_CLASS___IDSIDQueryController, "sharedInstance");
  v6 = (void *)objc_claimAutoreleasedReturnValue();
  +[IDSIDInfoOptions refreshIDInfo](&OBJC_CLASS___IDSIDInfoOptions, "refreshIDInfo");
  v7 = (void *)objc_claimAutoreleasedReturnValue();
  if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    LogPrintF();
  objc_msgSend(v6, "idInfoForDestinations:service:infoTypes:options:listenerID:queue:completionBlock:", v5, CFSTR("com.apple.private.alloy.nearby.family"), 1LL, v7, CFSTR("com.apple.private.alloy.nearby"), self->_dispatchQueue, v8);
  objc_release(v7);
  objc_release(v6);
  objc_release(v5);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(v8);
}

- (void)setIdsFamilyEndpointMap:(id)a3 
{
  NSDictionary *v4;
  NSDictionary *idsFamilyEndpointMap;
  uint64_t vars8;

  v4 = (NSDictionary *)objc_retain(a3);
  dispatch_assert_queue_V2((dispatch_queue_t)self->_dispatchQueue);
  idsFamilyEndpointMap = self->_idsFamilyEndpointMap;
  self->_idsFamilyEndpointMap = v4;
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(idsFamilyEndpointMap);
}

- (void)idsFamilyEndpointsUpdateWithForce:(BOOL)a3 
{
  _BOOL4 v3;
  void *v5;
  void *v6;
  double v7;
  double v8;
  _QWORD v9[5];

  v3 = a3;
  +[NSDate date](&OBJC_CLASS___NSDate, "date");
  v5 = (void *)objc_claimAutoreleasedReturnValue();
  v6 = v5;
  if (self->_lastFamilyQueryDate)
  {
    objc_msgSend(v5, "timeIntervalSinceDate:");
    v8 = v7;
  }
  else
  {
    v8 = 0.0;
  }
  if (-[RPCloudDaemon getDeviceClass](self, "getDeviceClass") == 4
    && !self->_idQueryInProgress
    && (v3 || v8 == 0.0 || v8 > 300.0) )
  {
    v9[0] = _NSConcreteStackBlock;
    v9[1] = 3221225472LL;
    v9[2] = sub_100011ED4;
    v9[3] = &unk_10010CCE0;
    v9[4] = self;
    -[RPCloudDaemon idsFamilyEndpointQueryWithCompletion:](self, "idsFamilyEndpointQueryWithCompletion:", v9);
    self->_idQueryInProgress = 1;
  }
  objc_release(v6);
}

- (int)idsHandheldCount
{
  RPCloudDaemon *v2;
  int idsHandheldCountCache;
  void *v4;
  id v5;
  uint64_t v6;
  void *i;
  void *v8;
  unsigned int v9;
  __int128 v11;
  __int128 v12;
  __int128 v13;
  __int128 v14;
  _BYTE v15[128];

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  idsHandheldCountCache = v2->_idsHandheldCountCache;
  if (idsHandheldCountCache < 0 && v2->_nearbyIDSService)
  {
    v2->_idsHandheldCountCache = 0;
    v11 = 0u;
    v12 = 0u;
    v13 = 0u;
    v14 = 0u;
    -[RPCloudDaemon idsDeviceArray](v2, "idsDeviceArray", 0LL);
    v4 = (void *)objc_claimAutoreleasedReturnValue();
    v5 = objc_msgSend(v4, "countByEnumeratingWithState:objects:count:", &v11, v15, 16LL);
    if (v5)
    {
      v6 = *(_QWORD *)v12;
      do
      {
        for (i = 0LL; i != v5; i = (char *)i + 1)
        {
          if (*(_QWORD *)v12 != v6)
            objc_enumerationMutation(v4);
          objc_msgSend(*(id *)(*((_QWORD *)&v11 + 1) + 8LL * (_QWORD)i), "modelIdentifier");
          v8 = (void *)objc_claimAutoreleasedReturnValue();
          v9 = GestaltProductTypeStringToDeviceClass() - 4;
          objc_release(v8);
          if (v9 >= 0xFFFFFFFD)
            ++v2->_idsHandheldCountCache;
        }
        v5 = objc_msgSend(v4, "countByEnumeratingWithState:objects:count:", &v11, v15, 16LL);
      }
      while (v5);
    }
    objc_release(v4);
    idsHandheldCountCache = v2->_idsHandheldCountCache;
  }
  objc_sync_exit(v2);
  objc_release(v2);
  return idsHandheldCountCache;
}

- (BOOL)idsHasAppleTV
{
  RPCloudDaemon *v2;
  int prefHasAppleTVForce;
  BOOL v4;
  void *v5;
  id v6;
  uint64_t v7;
  void *i;
  id v9;
  unsigned __int8 v10;
  __int128 v12;
  __int128 v13;
  __int128 v14;
  __int128 v15;
  _BYTE v16[128];

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  prefHasAppleTVForce = v2->_prefHasAppleTVForce;
  if (prefHasAppleTVForce < 0)
  {
    if (v2->_idsHasAppleTVCache < 0 && v2->_nearbyIDSService)
    {
      v2->_idsHasAppleTVCache = 0;
      v12 = 0u;
      v13 = 0u;
      v14 = 0u;
      v15 = 0u;
      -[RPCloudDaemon idsDeviceArray](v2, "idsDeviceArray", 0LL);
      v5 = (void *)objc_claimAutoreleasedReturnValue();
      v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
      if (v6)
      {
        v7 = *(_QWORD *)v13;
        while (2)
        {
          for (i = 0LL; i != v6; i = (char *)i + 1)
          {
            if (*(_QWORD *)v13 != v7)
              objc_enumerationMutation(v5);
            objc_msgSend(*(id *)(*((_QWORD *)&v12 + 1) + 8LL * (_QWORD)i), "modelIdentifier");
            v9 = objc_retain((id)objc_claimAutoreleasedReturnValue());
            v10 = (unsigned __int8)objc_msgSend(v9, "hasPrefix:", CFSTR("AppleTV"));
            objc_release(v9);
            objc_release(v9);
            if ((v10 & 1) != 0)
            {
              v2->_idsHasAppleTVCache = 1;
              goto LABEL_15;
            }
          }
          v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
          if (v6)
            continue;
          break;
        }
      }
LABEL_15:
      objc_release(v5);
    }
    v4 = v2->_idsHasAppleTVCache > 0;
  }
  else
  {
    v4 = prefHasAppleTVForce != 0;
  }
  objc_sync_exit(v2);
  objc_release(v2);
  return v4;
}

- (BOOL)idsHasHomePod
{
  RPCloudDaemon *v2;
  int prefHasHomePodForce;
  BOOL v4;
  void *v5;
  id v6;
  uint64_t v7;
  void *i;
  id v9;
  __int128 v11;
  __int128 v12;
  __int128 v13;
  __int128 v14;
  _BYTE v15[128];

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  prefHasHomePodForce = v2->_prefHasHomePodForce;
  if (prefHasHomePodForce < 0)
  {
    if (v2->_idsHasHomePodCache < 0 && v2->_nearbyIDSService)
    {
      v2->_idsHasHomePodCache = 0;
      v11 = 0u;
      v12 = 0u;
      v13 = 0u;
      v14 = 0u;
      -[RPCloudDaemon idsDeviceArray](v2, "idsDeviceArray", 0LL);
      v5 = (void *)objc_claimAutoreleasedReturnValue();
      v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v11, v15, 16LL);
      if (v6)
      {
        v7 = *(_QWORD *)v12;
        while (2)
        {
          for (i = 0LL; i != v6; i = (char *)i + 1)
          {
            if (*(_QWORD *)v12 != v7)
              objc_enumerationMutation(v5);
            objc_msgSend(*(id *)(*((_QWORD *)&v11 + 1) + 8LL * (_QWORD)i), "modelIdentifier");
            v9 = objc_retain((id)objc_claimAutoreleasedReturnValue());
            if (((unsigned int)objc_msgSend(v9, "hasPrefix:", CFSTR("AudioAccessory1,")) & 1) != 0
              || ((unsigned int)objc_msgSend(v9, "hasPrefix:", CFSTR("AudioAccessory5,")) & 1) != 0
              || ((unsigned int)objc_msgSend(v9, "hasPrefix:", CFSTR("AudioAccessory6,")) & 1) != 0
              || ((unsigned int)objc_msgSend(v9, "hasPrefix:", CFSTR("HomePod")) & 1) != 0 )
            {
              objc_release(v9);
              objc_release(v9);
              v2->_idsHasHomePodCache = 1;
              goto LABEL_18;
            }
            objc_release(v9);
            objc_release(v9);
          }
          v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v11, v15, 16LL);
          if (v6)
            continue;
          break;
        }
      }
LABEL_18:
      objc_release(v5);
    }
    v4 = v2->_idsHasHomePodCache > 0;
  }
  else
  {
    v4 = prefHasHomePodForce != 0;
  }
  objc_sync_exit(v2);
  objc_release(v2);
  return v4;
}

- (BOOL)idsHasiPad
{
  RPCloudDaemon *v2;
  int prefHasiPadForce;
  BOOL v4;
  void *v5;
  id v6;
  uint64_t v7;
  void *i;
  id v9;
  unsigned __int8 v10;
  __int128 v12;
  __int128 v13;
  __int128 v14;
  __int128 v15;
  _BYTE v16[128];

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  prefHasiPadForce = v2->_prefHasiPadForce;
  if (prefHasiPadForce < 0)
  {
    if (v2->_idsHasiPadCache < 0 && v2->_nearbyIDSService)
    {
      v2->_idsHasiPadCache = 0;
      v12 = 0u;
      v13 = 0u;
      v14 = 0u;
      v15 = 0u;
      -[RPCloudDaemon idsDeviceArray](v2, "idsDeviceArray", 0LL);
      v5 = (void *)objc_claimAutoreleasedReturnValue();
      v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
      if (v6)
      {
        v7 = *(_QWORD *)v13;
        while (2)
        {
          for (i = 0LL; i != v6; i = (char *)i + 1)
          {
            if (*(_QWORD *)v13 != v7)
              objc_enumerationMutation(v5);
            objc_msgSend(*(id *)(*((_QWORD *)&v12 + 1) + 8LL * (_QWORD)i), "modelIdentifier");
            v9 = objc_retain((id)objc_claimAutoreleasedReturnValue());
            v10 = (unsigned __int8)objc_msgSend(v9, "hasPrefix:", CFSTR("iPad"));
            objc_release(v9);
            objc_release(v9);
            if ((v10 & 1) != 0)
            {
              v2->_idsHasiPadCache = 1;
              goto LABEL_15;
            }
          }
          v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
          if (v6)
            continue;
          break;
        }
      }
LABEL_15:
      objc_release(v5);
    }
    v4 = v2->_idsHasiPadCache > 0;
  }
  else
  {
    v4 = prefHasiPadForce != 0;
  }
  objc_sync_exit(v2);
  objc_release(v2);
  return v4;
}

- (BOOL)idsHasMac
{
  RPCloudDaemon *v2;
  int prefHasMacForce;
  BOOL v4;
  void *v5;
  id v6;
  uint64_t v7;
  void *i;
  id v9;
  unsigned __int8 v10;
  __int128 v12;
  __int128 v13;
  __int128 v14;
  __int128 v15;
  _BYTE v16[128];

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  prefHasMacForce = v2->_prefHasMacForce;
  if (prefHasMacForce < 0)
  {
    if (v2->_idsHasMacCache < 0 && v2->_nearbyIDSService)
    {
      v2->_idsHasMacCache = 0;
      v12 = 0u;
      v13 = 0u;
      v14 = 0u;
      v15 = 0u;
      -[RPCloudDaemon idsDeviceArray](v2, "idsDeviceArray", 0LL);
      v5 = (void *)objc_claimAutoreleasedReturnValue();
      v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
      if (v6)
      {
        v7 = *(_QWORD *)v13;
        while (2)
        {
          for (i = 0LL; i != v6; i = (char *)i + 1)
          {
            if (*(_QWORD *)v13 != v7)
              objc_enumerationMutation(v5);
            objc_msgSend(*(id *)(*((_QWORD *)&v12 + 1) + 8LL * (_QWORD)i), "modelIdentifier");
            v9 = objc_retain((id)objc_claimAutoreleasedReturnValue());
            v10 = (unsigned __int8)objc_msgSend(v9, "containsString:", CFSTR("Mac"));
            objc_release(v9);
            objc_release(v9);
            if ((v10 & 1) != 0)
            {
              v2->_idsHasMacCache = 1;
              goto LABEL_15;
            }
          }
          v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
          if (v6)
            continue;
          break;
        }
      }
LABEL_15:
      objc_release(v5);
    }
    v4 = v2->_idsHasMacCache > 0;
  }
  else
  {
    v4 = prefHasMacForce != 0;
  }
  objc_sync_exit(v2);
  objc_release(v2);
  return v4;
}

- (BOOL)idsHasRealityDevice
{
  RPCloudDaemon *v2;
  int prefHasRealityDeviceForce;
  BOOL v4;
  void *v5;
  id v6;
  uint64_t v7;
  void *i;
  id v9;
  unsigned __int8 v10;
  __int128 v12;
  __int128 v13;
  __int128 v14;
  __int128 v15;
  _BYTE v16[128];

  v2 = objc_retain(self);
  objc_sync_enter(v2);
  prefHasRealityDeviceForce = v2->_prefHasRealityDeviceForce;
  if (prefHasRealityDeviceForce < 0)
  {
    if (v2->_idsHasRealityDeviceCache < 0 && v2->_nearbyIDSService)
    {
      v2->_idsHasRealityDeviceCache = 0;
      v12 = 0u;
      v13 = 0u;
      v14 = 0u;
      v15 = 0u;
      -[RPCloudDaemon idsDeviceArray](v2, "idsDeviceArray", 0LL);
      v5 = (void *)objc_claimAutoreleasedReturnValue();
      v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
      if (v6)
      {
        v7 = *(_QWORD *)v13;
        while (2)
        {
          for (i = 0LL; i != v6; i = (char *)i + 1)
          {
            if (*(_QWORD *)v13 != v7)
              objc_enumerationMutation(v5);
            objc_msgSend(*(id *)(*((_QWORD *)&v12 + 1) + 8LL * (_QWORD)i), "modelIdentifier");
            v9 = objc_retain((id)objc_claimAutoreleasedReturnValue());
            v10 = (unsigned __int8)objc_msgSend(v9, "hasPrefix:", CFSTR("RealityDevice"));
            objc_release(v9);
            objc_release(v9);
            if ((v10 & 1) != 0)
            {
              v2->_idsHasRealityDeviceCache = 1;
              goto LABEL_15;
            }
          }
          v6 = objc_msgSend(v5, "countByEnumeratingWithState:objects:count:", &v12, v16, 16LL);
          if (v6)
            continue;
          break;
        }
      }
LABEL_15:
      objc_release(v5);
    }
    v4 = v2->_idsHasRealityDeviceCache > 0;
  }
  else
  {
    v4 = prefHasRealityDeviceForce != 0;
  }
  objc_sync_exit(v2);
  objc_release(v2);
  return v4;
}

- (id)_idsURIWithID:(id)a3 
{
  id v3;
  void *v4;
  id v5;
  uint64_t v6;
  void *v7;
  void *v8;
  uint64_t vars8;

  v3 = objc_retain(a3);
  v4 = v3;
  if (v3)
  {
    if (objc_msgSend(v3, "_FZIDType") == (id)-1LL)
    {
      objc_msgSend(v4, "_bestGuessURI");
      v6 = objc_claimAutoreleasedReturnValue();
      v7 = (void *)v6;
      if (v6)
        v8 = (void *)v6;
      else
        v8 = v4;
      v5 = objc_retain(v8);
      objc_release(v7);
    }
    else
    {
      v5 = objc_retain(v4);
    }
  }
  else
  {
    v5 = 0LL;
  }
  objc_release(v4);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  return objc_autoreleaseReturnValue(v5);
}

- (void)service:(id)a3 account:(id)a4 incomingData:(id)a5 fromID:(id)a6 context:(id)a7 
{
  id v11;
  id v12;
  NSObject *dispatchQueue;
  IDSService *v14;
  char *v15;
  IDSService *nearbyIDSService;
  __CFString *v17;
  unsigned __int8 *v18;
  char *v19;
  unsigned __int8 *v20;
  unint64_t v21;
  RPCloudMessageContext *v22;
  void *v23;
  uint64_t v24;
  void *v25;
  char v26;
  unsigned __int8 *v27;
  id v28;
  uint64_t vars8;

  v28 = objc_retain(a5);
  v11 = objc_retain(a6);
  v12 = objc_retain(a7);
  dispatchQueue = (NSObject *)self->_dispatchQueue;
  v14 = (IDSService *)objc_retain(a3);
  dispatch_assert_queue_V2(dispatchQueue);
  v15 = (char *)objc_msgSend(v28, "length");
  nearbyIDSService = self->_nearbyIDSService;
  objc_release(v14);
  if (nearbyIDSService == v14)
  {
    v17 = objc_retain(CFSTR("com.apple.private.alloy.nearby"));
    v18 = (unsigned __int8 *)objc_msgSend(objc_retainAutorelease(v28), "bytes");
    v19 = v15 - 4;
    if ((unint64_t)v15 < 4)
    {
      if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
        goto LABEL_15;
      goto LABEL_12;
    }
    v20 = v18;
    v21 = ((unint64_t)v18[1] << 16) | ((unint64_t)v18[2] << 8) | v18[3];
    if ((unint64_t)v19 < v21)
    {
      if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
        goto LABEL_15;
LABEL_12:
      LogPrintF();
      goto LABEL_15;
    }
    v22 = objc_alloc_init(&OBJC_CLASS___RPCloudMessageContext);
    -[RPCloudMessageContext setCloudServiceID:](v22, "setCloudServiceID:", v17);
    -[RPCloudMessageContext setFromID:](v22, "setFromID:", v11);
    objc_msgSend(v12, "toID");
    v23 = (void *)objc_claimAutoreleasedReturnValue();
    -[RPCloudMessageContext setToID:](v22, "setToID:", v23);
    objc_release(v23);
    v24 = *v20;
    if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    {
      -[RPCloudMessageContext toID](v22, "toID");
      v25 = (void *)objc_claimAutoreleasedReturnValue();
      LogPrintF();
      objc_release(v25);
    }
    v26 = v24 - 32;
    if ((unsigned int)(v24 - 32) <= 0x21)
    {
      v27 = v20 + 4;
      if (((1LL << v26) & 7) != 0)
      {
        -[RPCloudDaemon _receivedFamilyIdentityFrameType:ptr:length:msgCtx:](self, "_receivedFamilyIdentityFrameType:ptr:length:msgCtx:", v24, v27, v21, v22);
        goto LABEL_38;
      }
      if (((1LL << v26) & 0x30000) != 0)
      {
        -[RPCloudDaemon _receivedWatchIdentityFrameType:ptr:length:fromID:](self, "_receivedWatchIdentityFrameType:ptr:length:fromID:", v24, v27, v21, v11);
        goto LABEL_38;
      }
      if (((1LL << v26) & 0x300000000LL) != 0)
      {
        -[RPCloudDaemon _receivedFriendIdentityFrameType:ptr:length:msgCtx:](self, "_receivedFriendIdentityFrameType:ptr:length:msgCtx:", v24, v27, v21, v22);
LABEL_38:
        objc_release(v22);
        goto LABEL_15;
      }
    }
    if ((_DWORD)v24 != 1 && dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    {
      sub_10001335C(v24);
      LogPrintF();
    }
    goto LABEL_38;
  }
  if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    LogPrintF();
  v17 = 0LL;
LABEL_15:
  objc_release(v17);
  objc_release(v12);
  objc_release(v11);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(v28);
}

- (void)service:(id)a3 account:(id)a4 identifier:(id)a5 didSendWithSuccess:(BOOL)a6 error:(id)a7 context:(id)a8 
{
  id v13;
  id v14;
  id v15;
  id v16;
  id v17;
  uint64_t vars8;

  v17 = objc_retain(a3);
  v13 = objc_retain(a4);
  v14 = objc_retain(a5);
  v15 = objc_retain(a7);
  v16 = objc_retain(a8);
  if (v15 || !a6)
  {
    if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      goto LABEL_8;
  }
  else if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
  {
LABEL_8:
    LogPrintF();
  }
  objc_release(v16);
  objc_release(v15);
  objc_release(v14);
  objc_release(v13);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(v17);
}

- (void)service:(id)a3 account:(id)a4 identifier:(id)a5 hasBeenDeliveredWithContext:(id)a6 
{
  id v9;
  id v10;
  id v11;
  id v12;
  uint64_t vars8;

  v12 = objc_retain(a3);
  v9 = objc_retain(a4);
  v10 = objc_retain(a5);
  v11 = objc_retain(a6);
  if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    LogPrintF();
  objc_release(v11);
  objc_release(v10);
  objc_release(v9);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(v12);
}

- (void)service:(id)a3 activeAccountsChanged:(id)a4 
{
  id v6;
  RPCloudDaemon *v7;
  NSArray *idsDeviceArray;
  NSDictionary *idsDeviceMap;
  void *v10;
  id v11;
  uint64_t vars8;

  v11 = objc_retain(a3);
  v6 = objc_retain(a4);
  if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
  {
    objc_msgSend(v6, "count");
    LogPrintF();
  }
  v7 = objc_retain(self);
  objc_sync_enter(v7);
  idsDeviceArray = v7->_idsDeviceArray;
  v7->_idsDeviceArray = 0LL;
  objc_release(idsDeviceArray);
  idsDeviceMap = v7->_idsDeviceMap;
  v7->_idsDeviceMap = 0LL;
  objc_release(idsDeviceMap);
  v7->_idsIsSignedInCache = -1;
  *(_QWORD *)&v7->_idsHasHomePodCache = -1LL;
  *(_QWORD *)&v7->_idsHasMacCache = -1LL;
  *(_QWORD *)&v7->_idsHandheldCountCache = -1LL;
  objc_sync_exit(v7);
  objc_release(v7);
  +[RPDaemon sharedDaemon](&OBJC_CLASS___RPDaemon, "sharedDaemon");
  v10 = (void *)objc_claimAutoreleasedReturnValue();
  objc_msgSend(v10, "postDaemonInfoChanges:", 1LL);
  objc_release(v10);
  objc_release(v6);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(v11);
}

- (void)service:(id)a3 devicesChanged:(id)a4 
{
  id v6;
  id v7;
  id v8;
  char *v9;
  char *v10;
  uint64_t v11;
  char *i;
  NSArray *v13;
  RPCloudDaemon *v14;
  NSArray *idsDeviceArray;
  NSArray *v16;
  NSDictionary *idsDeviceMap;
  void *v18;
  __int128 v19;
  __int128 v20;
  __int128 v21;
  __int128 v22;
  _BYTE v23[128];

  v6 = objc_retain(a3);
  v7 = objc_retain(a4);
  if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
  {
    objc_msgSend(v7, "count");
    LogPrintF();
  }
  v21 = 0u;
  v22 = 0u;
  v19 = 0u;
  v20 = 0u;
  v8 = objc_retain(v7);
  v9 = (char *)objc_msgSend(v8, "countByEnumeratingWithState:objects:count:", &v19, v23, 16LL);
  if (v9)
  {
    v10 = v9;
    v11 = *(_QWORD *)v20;
    while (2)
    {
      for (i = 0LL; i != v10; ++i)
      {
        if (*(_QWORD *)v20 != v11)
          objc_enumerationMutation(v8);
        objc_opt_class();
        if ((objc_opt_isKindOfClass() & 1) == 0)
        {
          v18 = v8;
          if (dword_10012BA38 <= 90)
          {
            if (dword_10012BA38 != -1 || (v18 = v8, (unsigned int)_LogCategory_Initialize()))
            {
              LogPrintF();
              v18 = v8;
            }
          }
          goto LABEL_18;
        }
      }
      v10 = (char *)objc_msgSend(v8, "countByEnumeratingWithState:objects:count:", &v19, v23, 16LL);
      if (v10)
        continue;
      break;
    }
  }
  objc_release(v8);
  v13 = (NSArray *)objc_msgSend(v8, "copy");
  v14 = objc_retain(self);
  objc_sync_enter(v14);
  idsDeviceArray = v14->_idsDeviceArray;
  v14->_idsDeviceArray = v13;
  v16 = objc_retain(v13);
  objc_release(idsDeviceArray);
  idsDeviceMap = v14->_idsDeviceMap;
  v14->_idsDeviceMap = 0LL;
  objc_release(idsDeviceMap);
  v14->_idsIsSignedInCache = -1;
  *(_QWORD *)&v14->_idsHasHomePodCache = -1LL;
  *(_QWORD *)&v14->_idsHasMacCache = -1LL;
  *(_QWORD *)&v14->_idsHandheldCountCache = -1LL;
  objc_sync_exit(v14);
  objc_release(v14);
  +[RPDaemon sharedDaemon](&OBJC_CLASS___RPDaemon, "sharedDaemon");
  v18 = (void *)objc_claimAutoreleasedReturnValue();
  objc_release(v16);
  objc_msgSend(v18, "postDaemonInfoChanges:", 1LL);
LABEL_18:
  objc_release(v18);
  objc_release(v8);
  objc_release(v6);
}

- (void)serviceSpaceDidBecomeAvailable:(id)a3 
{
  id v3;
  void *v4;
  int v5;
  id v6;
  uint64_t vars8;

  v3 = objc_retain(a3);
  v4 = v3;
  if (dword_10012BA38 <= 30)
  {
    v6 = v3;
    if (dword_10012BA38 != -1 || (v5 = _LogCategory_Initialize(), v4 = v6, v5))
    {
      LogPrintF();
      v4 = v6;
    }
  }
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(v4);
}

- (void)_receivedFamilyIdentityFrameType:(unsigned __int8)a3 ptr:(const char *)a4 length:(unint64_t)a5 msgCtx:(id)a6 
{
  int v6;
  id v7;
  void *v8;
  void *v9;
  void *v10;
  void *v11;
  void *v12;
  void *v13;

  v6 = a3;
  v7 = objc_retain(a6);
  objc_msgSend(v7, "fromID");
  v8 = (void *)objc_claimAutoreleasedReturnValue();
  v9 = (void *)OPACKDecodeBytes();
  if (v9)
  {
    objc_opt_class();
    if ((objc_opt_isKindOfClass() & 1) != 0)
    {
      v10 = (void *)IDSCopyRawAddressForDestination();
      v11 = v10;
      if (v10)
      {
        if ((unsigned int)objc_msgSend(v10, "_appearsToBePhoneNumber"))
          CUNormalizePhoneNumber();
        else
          CUNormalizeEmailAddress();
        v12 = (void *)objc_claimAutoreleasedReturnValue();
        objc_msgSend(v7, "setAppleID:", v12);
        objc_release(v12);
        switch(v6)
        {
          case '"':
            +[RPPeopleDaemon sharedPeopleDaemon](&OBJC_CLASS___RPPeopleDaemon, "sharedPeopleDaemon");
            v13 = (void *)objc_claimAutoreleasedReturnValue();
            objc_msgSend(v13, "receivedFamilyIdentityUpdate:msgCtx:", v9, v7);
            goto LABEL_26;
          case '!':
            +[RPPeopleDaemon sharedPeopleDaemon](&OBJC_CLASS___RPPeopleDaemon, "sharedPeopleDaemon");
            v13 = (void *)objc_claimAutoreleasedReturnValue();
            objc_msgSend(v13, "receivedFamilyIdentityResponse:msgCtx:", v9, v7);
            goto LABEL_26;
          case ' ':
            +[RPPeopleDaemon sharedPeopleDaemon](&OBJC_CLASS___RPPeopleDaemon, "sharedPeopleDaemon");
            v13 = (void *)objc_claimAutoreleasedReturnValue();
            objc_msgSend(v13, "receivedFamilyIdentityRequest:msgCtx:", v9, v7);
LABEL_26:
            objc_release(v13);
LABEL_34:
            objc_release(v11);
            goto LABEL_36;
        }
        if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
          goto LABEL_34;
        sub_10001335C(v6);
      }
      else if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
      {
        goto LABEL_34;
      }
      LogPrintF();
      goto LABEL_34;
    }
    if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
LABEL_35:
      LogPrintF();
  }
  else if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
  {
    goto LABEL_35;
  }
LABEL_36:
  objc_release(v9);
  objc_release(v8);
  objc_release(v7);
}

- (void)_receivedFriendIdentityFrameType:(unsigned __int8)a3 ptr:(const char *)a4 length:(unint64_t)a5 msgCtx:(id)a6 
{
  int v6;
  id v7;
  void *v8;
  void *v9;
  void *v10;
  void *v11;
  void *v12;
  void *v13;

  v6 = a3;
  v7 = objc_retain(a6);
  objc_msgSend(v7, "fromID");
  v8 = (void *)objc_claimAutoreleasedReturnValue();
  v9 = (void *)OPACKDecodeBytes();
  if (v9)
  {
    objc_opt_class();
    if ((objc_opt_isKindOfClass() & 1) != 0)
    {
      v10 = (void *)IDSCopyRawAddressForDestination();
      v11 = v10;
      if (v10)
      {
        if ((unsigned int)objc_msgSend(v10, "_appearsToBePhoneNumber"))
          CUNormalizePhoneNumber();
        else
          CUNormalizeEmailAddress();
        v12 = (void *)objc_claimAutoreleasedReturnValue();
        objc_msgSend(v7, "setAppleID:", v12);
        objc_release(v12);
        switch(v6)
        {
          case 'B':
            +[RPPeopleDaemon sharedPeopleDaemon](&OBJC_CLASS___RPPeopleDaemon, "sharedPeopleDaemon");
            v13 = (void *)objc_claimAutoreleasedReturnValue();
            objc_msgSend(v13, "receivedFriendIdentityUpdate:msgCtx:", v9, v7);
            goto LABEL_26;
          case 'A':
            +[RPPeopleDaemon sharedPeopleDaemon](&OBJC_CLASS___RPPeopleDaemon, "sharedPeopleDaemon");
            v13 = (void *)objc_claimAutoreleasedReturnValue();
            objc_msgSend(v13, "receivedFriendIdentityResponse:msgCtx:", v9, v7);
            goto LABEL_26;
          case '@':
            +[RPPeopleDaemon sharedPeopleDaemon](&OBJC_CLASS___RPPeopleDaemon, "sharedPeopleDaemon");
            v13 = (void *)objc_claimAutoreleasedReturnValue();
            objc_msgSend(v13, "receivedFriendIdentityRequest:msgCtx:", v9, v7);
LABEL_26:
            objc_release(v13);
LABEL_34:
            objc_release(v11);
            goto LABEL_36;
        }
        if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
          goto LABEL_34;
        sub_10001335C(v6);
      }
      else if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
      {
        goto LABEL_34;
      }
      LogPrintF();
      goto LABEL_34;
    }
    if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
LABEL_35:
      LogPrintF();
  }
  else if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
  {
    goto LABEL_35;
  }
LABEL_36:
  objc_release(v9);
  objc_release(v8);
  objc_release(v7);
}

- (void)_receivedWatchIdentityFrameType:(unsigned __int8)a3 ptr:(const char *)a4 length:(unint64_t)a5 fromID:(id)a6 
{
  int v6;
  id v8;
  void *v9;
  void *v10;
  void *v11;
  void *v12;

  v6 = a3;
  v8 = objc_retain(a6);
  if (((unsigned int)objc_msgSend(v8, "hasPrefix:", CFSTR("device:")) & 1) != 0)
  {
    -[RPCloudDaemon idsDeviceMap](self, "idsDeviceMap");
    v9 = (void *)objc_claimAutoreleasedReturnValue();
    objc_msgSend(v8, "substringFromIndex:", 7LL);
    v10 = (void *)objc_claimAutoreleasedReturnValue();
    objc_msgSend(v9, "objectForKeyedSubscript:", v10);
    v11 = (void *)objc_claimAutoreleasedReturnValue();
    objc_release(v10);
    objc_release(v9);
    if (!v11)
    {
      if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
        LogPrintF();
      goto LABEL_36;
    }
    v12 = (void *)OPACKDecodeBytes();
    if (v12)
    {
      objc_opt_class();
      if ((objc_opt_isKindOfClass() & 1) != 0)
      {
        if (v6 == 49)
        {
          -[RPCloudDaemon _receivedWatchIdentityResponse:fromIDSDevice:](self, "_receivedWatchIdentityResponse:fromIDSDevice:", v12, v11);
          goto LABEL_35;
        }
        if (v6 == 48)
        {
          -[RPCloudDaemon _receivedWatchIdentityRequest:fromIDSDevice:](self, "_receivedWatchIdentityRequest:fromIDSDevice:", v12, v11);
LABEL_35:
          objc_release(v12);
LABEL_36:
          objc_release(v11);
          goto LABEL_37;
        }
        if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
          goto LABEL_35;
        sub_10001335C(v6);
      }
      else if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
      {
        goto LABEL_35;
      }
    }
    else if (dword_10012BA38 > 90 || dword_10012BA38 == -1 && !(unsigned int)_LogCategory_Initialize())
    {
      goto LABEL_35;
    }
    LogPrintF();
    goto LABEL_35;
  }
  if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    LogPrintF();
LABEL_37:
  objc_release(v8);
}

- (void)_receivedWatchIdentityRequest:(id)a3 fromIDSDevice:(id)a4 
{
  id v6;
  void *v7;
  void *v8;
  id v9;
  void *v10;
  void *v11;
  id v12;
  uint64_t vars8;

  v12 = objc_retain(a3);
  v6 = objc_retain(a4);
  dispatch_assert_queue_V2((dispatch_queue_t)self->_dispatchQueue);
  if (((unsigned int)-[CUSystemMonitor firstUnlocked](self->_systemMonitor, "firstUnlocked") & 1) != 0)
  {
    objc_msgSend(v6, "uniqueIDOverride");
    v7 = (void *)objc_claimAutoreleasedReturnValue();
    if (v7)
    {
      if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      {
        v11 = v7;
        LogPrintF();
      }
      +[RPIdentityDaemon sharedIdentityDaemon](&OBJC_CLASS___RPIdentityDaemon, "sharedIdentityDaemon", v11);
      v8 = (void *)objc_claimAutoreleasedReturnValue();
      objc_msgSend(v8, "saveIdentityWithIDSDeviceID:message:error:", v7, v12, 0LL);
      objc_release(v8);
      v9 = objc_alloc_init((Class)&OBJC_CLASS___NSMutableDictionary);
      +[RPIdentityDaemon sharedIdentityDaemon](&OBJC_CLASS___RPIdentityDaemon, "sharedIdentityDaemon");
      v10 = (void *)objc_claimAutoreleasedReturnValue();
      objc_msgSend(v10, "addSelfIdentityInfoToMessage:flags:", v9, 0LL);
      objc_release(v10);
      -[RPCloudDaemon sendIDSMessage:cloudServiceID:frameType:destinationDevice:sendFlags:msgCtx:error:](self, "sendIDSMessage:cloudServiceID:frameType:destinationDevice:sendFlags:msgCtx:error:", v9, CFSTR("com.apple.private.alloy.nearby"), 49LL, v6, 1LL, 0LL, 0LL);
      objc_release(v9);
    }
    else if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    {
      LogPrintF();
    }
    objc_release(v7);
  }
  else if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
  {
    LogPrintF();
  }
  objc_release(v6);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(v12);
}

- (void)_receivedWatchIdentityResponse:(id)a3 fromIDSDevice:(id)a4 
{
  id v6;
  void *v7;
  void *v8;
  void *v9;
  id v10;
  uint64_t vars8;

  v10 = objc_retain(a3);
  v6 = objc_retain(a4);
  dispatch_assert_queue_V2((dispatch_queue_t)self->_dispatchQueue);
  if (((unsigned int)-[CUSystemMonitor firstUnlocked](self->_systemMonitor, "firstUnlocked") & 1) != 0)
  {
    objc_msgSend(v6, "uniqueIDOverride");
    v7 = (void *)objc_claimAutoreleasedReturnValue();
    if (v7)
    {
      if (dword_10012BA38 <= 30 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      {
        v9 = v7;
        LogPrintF();
      }
      +[RPIdentityDaemon sharedIdentityDaemon](&OBJC_CLASS___RPIdentityDaemon, "sharedIdentityDaemon", v9);
      v8 = (void *)objc_claimAutoreleasedReturnValue();
      objc_msgSend(v8, "saveIdentityWithIDSDeviceID:message:error:", v7, v10, 0LL);
      objc_release(v8);
    }
    else if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    {
      LogPrintF();
    }
    objc_release(v7);
  }
  else if (dword_10012BA38 <= 60 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
  {
    LogPrintF();
  }
  objc_release(v6);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_release(v10);
}

- (BOOL)sendIDSMessage:(id)a3 cloudServiceID:(id)a4 frameType:(unsigned __int8)a5 destinationDevice:(id)a6 sendFlags:(unsigned int)a7 msgCtx:(id)a8 error:(id *)a9 
{
  uint64_t v10;
  uint64_t v12;
  id v15;
  id v16;
  id v17;
  id v18;
  void *v19;
  BOOL v20;
  void *v21;

  v10 = *(_QWORD *)&a7;
  v12 = a5;
  v15 = objc_retain(a3);
  v16 = objc_retain(a4);
  v17 = objc_retain(a6);
  v18 = objc_retain(a8);
  v19 = (void *)IDSCopyIDForDevice();
  if (v19)
  {
    v20 = -[RPCloudDaemon sendIDSMessage:cloudServiceID:frameType:destinationID:sendFlags:msgCtx:error:](self, "sendIDSMessage:cloudServiceID:frameType:destinationID:sendFlags:msgCtx:error:", v15, v16, v12, v19, v10, v18, a9);
  }
  else
  {
    RPErrorF();
    v21 = (void *)objc_claimAutoreleasedReturnValue();
    if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
      LogPrintF();
    if (a9)
      *a9 = objc_retainAutorelease(v21);
    objc_release(v21);
    v20 = 0;
  }
  objc_release(v19);
  objc_release(v18);
  objc_release(v17);
  objc_release(v16);
  objc_release(v15);
  return v20;
}

- (BOOL)sendIDSMessage:(id)a3 cloudServiceID:(id)a4 frameType:(unsigned __int8)a5 destinationID:(id)a6 sendFlags:(unsigned int)a7 msgCtx:(id)a8 error:(id *)a9 
{
  char v10;
  id v14;
  __CFString *v15;
  id v16;
  void *Data;
  void *v18;
  id v19;
  id v20;
  id v21;
  void *v22;
  uint64_t v23;
  void *v24;
  void *v25;
  void *v26;
  void *v27;
  IDSService *v28;
  id v29;
  void *v30;
  uint64_t v31;
  BOOL v32;
  void *v33;
  unsigned __int8 v34;
  void *v35;
  void *v37;
  id v38;
  void *v39;
  id v40;
  id v41;
  void *v42;
  id v43;
  void *v44;
  id v45;
  id v46;
  _BYTE v47[4];
  int v48;

  v10 = a7;
  v14 = objc_retain(a4);
  v15 = (__CFString *)objc_retain(a6);
  v16 = objc_retain(a8);
  v48 = 0;
  Data = (void *)OPACKEncoderCreateData();
  v18 = Data;
  if (Data)
  {
    v41 = v14;
    v47[0] = a5;
    v19 = objc_msgSend(Data, "length");
    v47[1] = BYTE2(v19);
    v47[2] = BYTE1(v19);
    v47[3] = (_BYTE)v19;
    v20 = objc_alloc_init((Class)&OBJC_CLASS___NSMutableData);
    objc_msgSend(v20, "appendBytes:length:", v47, 4LL);
    objc_msgSend(v20, "appendData:", v18);
    v21 = objc_alloc_init((Class)&OBJC_CLASS___NSMutableDictionary);
    v22 = v21;
    if ((v10 & 1) != 0)
    {
      objc_msgSend(v21, "setObject:forKeyedSubscript:", &__kCFBooleanTrue, IDSSendMessageOptionLocalDeliveryKey);
      objc_msgSend(v22, "setObject:forKeyedSubscript:", &__kCFBooleanTrue, IDSSendMessageOptionRequireBluetoothKey);
    }
    v23 = IDSSendMessageOptionFromIDKey;
    objc_msgSend(v16, "toID");
    v24 = (void *)objc_claimAutoreleasedReturnValue();
    if (v24)
    {
      objc_msgSend(v22, "setObject:forKeyedSubscript:", v24, v23);
      -[RPCloudDaemon _idsAccountWithURI:senderID:](self, "_idsAccountWithURI:senderID:", v24, 0LL);
      v25 = (void *)objc_claimAutoreleasedReturnValue();
      objc_release(v24);
      v14 = v41;
      if (v25)
        goto LABEL_16;
    }
    else
    {
      objc_release(0LL);
      v14 = v41;
    }
    objc_msgSend(v16, "sendersKnownAlias");
    v27 = (void *)objc_claimAutoreleasedReturnValue();
    -[RPCloudDaemon _idsURIWithID:](self, "_idsURIWithID:", v27);
    v46 = (id)objc_claimAutoreleasedReturnValue();
    objc_release(v27);
    if (v46)
    {
      -[RPCloudDaemon _idsAccountWithURI:senderID:](self, "_idsAccountWithURI:senderID:", v46, &v46);
      v25 = (void *)objc_claimAutoreleasedReturnValue();
      if (v25)
        objc_msgSend(v22, "setObject:forKeyedSubscript:", v46, v23);
    }
    else
    {
      v25 = 0LL;
    }
    objc_release(v46);
LABEL_16:
    if ((unsigned int)objc_msgSend(v16, "nonWakingRequest"))
      objc_msgSend(v22, "setObject:forKeyedSubscript:", &off_100116040, IDSSendMessageOptionPushPriorityKey);
    if (!(unsigned int)objc_msgSend(v14, "isEqual:", CFSTR("com.apple.private.alloy.nearby"))
      || (v28 = objc_retain(self->_nearbyIDSService)) == 0LL )
    {
      RPErrorF();
      v29 = (id)objc_claimAutoreleasedReturnValue();
      if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
        LogPrintF();
      if (a9)
      {
        v29 = objc_retainAutorelease(v29);
        v32 = 0;
        *a9 = v29;
      }
      else
      {
        v32 = 0;
      }
      goto LABEL_59;
    }
    v29 = v28;
    v15 = objc_retain(v15);
    v30 = v15;
    if (((unsigned int)-[__CFString hasPrefix:](v15, "hasPrefix:", CFSTR("token:")) & 1) == 0)
    {
      if ((unsigned int)-[__CFString _appearsToBePhoneNumber](v15, "_appearsToBePhoneNumber"))
      {
        v31 = IDSCopyIDForPhoneNumber();
LABEL_36:
        v30 = (void *)v31;
        objc_release(v15);
        goto LABEL_37;
      }
      v30 = v15;
      if ((unsigned int)-[__CFString _appearsToBeEmail](v15, "_appearsToBeEmail"))
      {
        v31 = IDSCopyIDForEmailAddress();
        goto LABEL_36;
      }
    }
LABEL_37:
    if ((unsigned int)objc_msgSend(v30, "isEqual:", v15))
    {
      objc_release(v15);
      v15 = CFSTR("=");
    }
    v39 = v30;
    if (v30)
    {
      +[NSSet setWithObject:](&OBJC_CLASS___NSSet, "setWithObject:", v30);
      v33 = (void *)objc_claimAutoreleasedReturnValue();
      if (v25)
      {
        v45 = 0LL;
        v44 = 0LL;
        v34 = (unsigned __int8)objc_msgSend(v29, "sendData:fromAccount:toDestinations:priority:options:identifier:error:", v20, v25, v33, 300LL, v22, &v45, &v44);
        v38 = objc_retain(v45);
        v35 = v44;
      }
      else
      {
        v42 = 0LL;
        v43 = 0LL;
        v34 = (unsigned __int8)objc_msgSend(v29, "sendData:toDestinations:priority:options:identifier:error:", v20, v33, 300LL, v22, &v43, &v42);
        v38 = objc_retain(v43);
        v35 = v42;
      }
      v40 = objc_retain(v35);
      objc_release(v33);
      if ((v34 & 1) != 0)
      {
        if (dword_10012BA38 > 30)
        {
          v32 = 1;
          v14 = v41;
        }
        else
        {
          v14 = v41;
          if (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize())
          {
            objc_msgSend(v25, "loginID");
            v37 = (void *)objc_claimAutoreleasedReturnValue();
            LogPrintF();
            objc_release(v37);
          }
          v32 = 1;
        }
      }
      else
      {
        if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
          LogPrintF();
        v14 = v41;
        if (a9)
        {
          RPNestedErrorF();
          v32 = 0;
          *a9 = objc_autorelease((id)objc_claimAutoreleasedReturnValue());
        }
        else
        {
          v32 = 0;
        }
      }
      objc_release(v40);
      objc_release(v38);
    }
    else if (a9)
    {
      RPErrorF();
      v32 = 0;
      *a9 = objc_autorelease((id)objc_claimAutoreleasedReturnValue());
    }
    else
    {
      v32 = 0;
    }
    objc_release(v39);
LABEL_59:
    objc_release(v29);
    objc_release(v25);
    objc_release(v22);
    objc_release(v20);
    goto LABEL_60;
  }
  RPErrorF();
  v26 = (void *)objc_claimAutoreleasedReturnValue();
  if (dword_10012BA38 <= 90 && (dword_10012BA38 != -1 || (unsigned int)_LogCategory_Initialize()))
    LogPrintF();
  if (a9)
    *a9 = objc_retainAutorelease(v26);
  objc_release(v26);
  v32 = 0;
LABEL_60:
  objc_release(v18);
  objc_release(v16);
  objc_release(v15);
  objc_release(v14);
  return v32;
}

- (OS_dispatch_queue)dispatchQueue
{
  return self->_dispatchQueue;
}

- (void)setDispatchQueue:(id)a3 
{
  objc_storeStrong((id *)&self->_dispatchQueue, a3);
}

- (IDSService)nearbyIDSService
{
  return self->_nearbyIDSService;
}

- (void)setNearbyIDSService:(id)a3 
{
  objc_storeStrong((id *)&self->_nearbyIDSService, a3);
}

- (void).cxx_destruct
{
  uint64_t vars8;

  objc_storeStrong((id *)&self->_nearbyIDSService, 0LL);
  objc_storeStrong((id *)&self->_dispatchQueue, 0LL);
  objc_storeStrong((id *)&self->_systemMonitor, 0LL);
  objc_storeStrong((id *)&self->_lastFamilyQueryDate, 0LL);
  objc_storeStrong((id *)&self->_idsFamilyEndpointMap, 0LL);
  objc_storeStrong((id *)&self->_idsDeviceMap, 0LL);
  objc_storeStrong((id *)&self->_idsDeviceIDSelf, 0LL);
  objc_storeStrong((id *)&self->_idsDeviceArray, 0LL);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_storeStrong((id *)&self->_idsCorrelationIdentifier, 0LL);
}

@end
