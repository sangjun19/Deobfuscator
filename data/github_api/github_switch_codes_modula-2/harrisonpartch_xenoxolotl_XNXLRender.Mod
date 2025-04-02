// Repository: harrisonpartch/xenoxolotl
// File: XNXLRender.Mod

MODULE XNXLRender;
IMPORT SYSTEM, Kernel, Modules,Objects, Random, XNXLMarsh, 
 Out := KernelLog, Math, Base:=XNXLBase,  XNXLMath,  XNXLRetina,Heaps,
	 X11,  Api := X11Api, GL:=OpenGL, GLC := OpenGLConst, Inputs, KS := X11KeySymDef, V := XF86VMode;

CONST PX=3; (* image size multiplier -- must be 3 for coorect foveation! *)
	ML = 0;  MM = 1;  MR = 2;
	Button1Bit = 8; Button2Bit = 9; Button3Bit =10;
	ShiftKeyBit = 0;  ShiftLokBit = 1;  ControlKeyBit = 2;  Mod1KeyBit = 3;
	Mod2KeyBit = 4;  Mod3KeyBit = 5;  Mod4KeyBit = 6;  Mod5KeyBit = 7;	  
	rad2deg=360/6.28318;
	
TYPE Aperture = Base.Aperture;
TYPE Ray = Base.Ray;
TYPE Voxel = Base.Voxel; 
TYPE PT = Base.PT;
TYPE Hints = RECORD
	flags: SET;
	functions: LONGINT;
	decorations: LONGINT;
	inputMode: LONGINT;
	status: LONGINT
END;
TYPE SNAP = RECORD
	lookphi, looktheta : REAL;
	aperture: Aperture;
	x,y,z,cdroll: REAL;		
	lookdxyz:PT;
END;	

TYPE Camera = OBJECT
VAR
	image,imj: Base.IMAGE;
	random: Random.Generator;
	rayschanged, ang1,filter: BOOLEAN;
	fovealeft, fovearight, foveabottom, foveatop: LONGINT;
	fovea: BOOLEAN;
	cam,worldcenter: PT; 
	d2c:REAL;
	W,H,pointerx,pointery:LONGINT;
	aperture: Aperture;
	iamalargecamera: BOOLEAN;
	lookdxyz:PT;
	cx, cy, cz, cvx, cvy, cvz, cvl, cvu: REAL;   
	FWard,xaxis,yaxis,zaxis:PT; 
	croll: REAL;
	cdroll: REAL;
	cameratheta,cameraphi: REAL;
	world:Voxel;	
	positionlock, orientationlock, returntohorizon: BOOLEAN;	
	fisheye: REAL;	

PROCEDURE & init (width, height: INTEGER);
BEGIN
	NEW(image);
	NEW(imj); 
	W := width; H := height;
	filter:=FALSE;
	cameratheta := 0;
	cameraphi := 0;
	aperture.width :=4/3;
	aperture.height :=3/3;
	Base.setPT(xaxis, 1,0,0);	
	Base.setPT(yaxis,0,1,0);	
	Base.setPT(zaxis,0,0,-1);	
	Base.setPT(worldcenter,1/2,1/2,1/2);	
	fisheye:=1;
	GLIDE:=0.98;
	hop; hop;
END init;

PROCEDURE hop;
BEGIN
	stop;
	cx :=1/2+Base.rand.Uniform()/2; cy:=1/2+Base.rand.Uniform()/2; cz := 1/2+Base.rand.Uniform()/2;
END hop;

PROCEDURE move;
VAR
	face: INTEGER;
	x,y,z,d: REAL;
	v, target: Voxel;
	p,normal: PT;
	proberay: Ray;
	pass,slower: BOOLEAN;

PROCEDURE normalize(VAR x,y,z,d: REAL);
BEGIN
	d := Math.sqrt(x*x + y*y+z*z);
	x := x/d; y := y/d; z:=z/d
END normalize;	

PROCEDURE denormalize(VAR x,y,z,d: REAL);
BEGIN
	x := x*d; y := y*d; z:=z*d
END denormalize;	

PROCEDURE reflect(VAR x,y,z: REAL; nx,ny,nz:REAL);
VAR 
	dot: REAL;
BEGIN
	dot := x*nx+y*ny+z*nz;
	nx := 2*nx*dot; ny := 2*ny*dot; nz := 2*nz*dot;
	x := x-nx; y := y-ny; z := z-nz; 
END reflect;

BEGIN
	proberay:= Base.rays[W DIV 2, H DIV 2];
	x := cx + cvx; y := cy + cvy; z := cz + cvz;
	p:=Base.mkPT(x,y,z);
	v := Base.world.probe(p,1000);
	pass := Base.world.passprobe(Base.mkPT(x,y,z));
	lookdxyz:=proberay.dxyz;
	IF ~pass THEN
		IF v#NIL THEN 
			v.play
		ELSE
			Base.world.play
		END; 
		x := cx + cvx; y := cy + cvy; z := cz;	
		pass := Base.world.passprobe(Base.mkPT(x,y,z));
		IF pass THEN 
			cvz:=-cvz/2
		ELSE
			x := cx + cvx; y := cy; z := cz+cvz;	
			pass := Base.world.passprobe(Base.mkPT(x,y,z));
			IF pass THEN 
				cvy:=0
			ELSE			
				x := cx; y := cy + cvy; z := cz+cvz;	
				pass := Base.world.passprobe(Base.mkPT(x,y,z));
				IF pass THEN 
					cvx:=0 
				END
			END
		END
	END;
	IF  pass THEN 
		cx:=x; cy:=y; cz:=z 
	ELSE
		Base.world.Shade(proberay);
		normalize(cvx,cvy,cvz,d);
		reflect(cvx, cvy, cvz, proberay.normal.x, proberay.normal.y, proberay.normal.z);
		denormalize(cvx,cvy,cvz,d);	
		cvx:=cvx/2; cvy:=cvy/2; cvz:=cvz/2; 
	END;
	IF cx<0 THEN cx:=cx+1
	ELSIF cx>1 THEN cx:=cx-1
	END;
	IF cy<0 THEN cy:=cy+1
	ELSIF cy>1 THEN cy:=cy-1
	END;
	IF cz<0 THEN cz:=cz+1
	ELSIF cz>1 THEN cz:=cz-1
	END;	
END move; 

PROCEDURE stop;
BEGIN
	Base.speed := Base.speed/1.05;
	cvx:=0;
	cvy:=0;
	cvz:=0;
END stop;

PROCEDURE glidetoggle;
BEGIN
	IF GLIDE<1/2 THEN GLIDE:=0.95 ELSE GLIDE:=0.3 END;
END glidetoggle;

PROCEDURE forward(x:REAL);
VAR
	fward,rward, down, down2, rward2, right:PT;
BEGIN
	fward:=xaxis;
	rward:=yaxis;
	down:=zaxis;
	XNXLMath.orrot(fward, zaxis, cameraphi);  
	XNXLMath.orrot(rward, zaxis, cameraphi);  
	XNXLMath.orrot(fward, rward, cameratheta);  
	XNXLMath.orrot(down, rward, cameratheta);  	
	cvx :=cvx+(x*fward.x * Base.speed); 
	cvy :=cvy+(x*fward.y * Base.speed); 	
	cvz :=cvz+(x*fward.z * Base.speed); 	
END forward;

PROCEDURE backward(x:REAL);
VAR	
	fward,rward, down, right:PT;
BEGIN
	fward:=xaxis;
	rward:=yaxis;
	down:=zaxis;
	XNXLMath.orrot(fward, zaxis, cameraphi);  
	XNXLMath.orrot(rward, zaxis, cameraphi);  
	XNXLMath.orrot(fward, rward, cameratheta);  
	XNXLMath.orrot(down, rward, cameratheta);  	
	cvx :=cvx-(x*fward.x * Base.speed); 
	cvy :=cvy-(x*fward.y * Base.speed); 	
	cvz :=cvz-(x*fward.z * Base.speed); 	
END backward;

PROCEDURE rightstrafe(x:REAL);
VAR
	rward:PT;
BEGIN
	rward:=Base.rays[Base.W-1,Base.H DIV 2].dxyz;
	cvx :=cvx+(x*rward.x * Base.speed); 
	cvy :=cvy+(x*rward.y * Base.speed); 	
	cvz :=cvz+(x*rward.z * Base.speed); 	
END rightstrafe;

PROCEDURE leftstrafe(x:REAL);
VAR
	lward:PT;
BEGIN
	lward:=Base.rays[0,Base.H DIV 2].dxyz;
	cvx :=cvx+(x*lward.x * Base.speed); 
	cvy :=cvy+(x*lward.y * Base.speed); 	
	cvz :=cvz+(x*lward.z * Base.speed); 	
END leftstrafe;

PROCEDURE upstrafe(x:REAL); (* strafe up perpendicular to look *)
VAR
	up:PT;
BEGIN
	up:=Base.rays[Base.W DIV 2,0].dxyz;
	cvx :=cvx+(x*up.x * Base.speed); 
	cvy :=cvy+(x*up.y * Base.speed); 	
	cvz :=cvz+(x*up.z * Base.speed); 	
END upstrafe;

PROCEDURE downstrafe(x:REAL);
VAR
	down:PT;
BEGIN
	down:=Base.rays[Base.W DIV 2,Base.H-1].dxyz;	
	cvx :=cvx+(x*down.x * Base.speed); 
	cvy :=cvy+(x*down.y * Base.speed); 	
	cvz :=cvz+(x*down.z * Base.speed); 	
END downstrafe;

PROCEDURE initrays;
VAR
	reversej, i, j: LONGINT;
	theta, phi, dtheta, dphi: REAL;
	lookperpray: Ray;
	lookvector:PT;
	look: Base.PT;
	camtweak: Base.PT;
	d1,d2,d3: REAL;
	w,h: REAL;
	fward,rward, down, down2, rward2, right:PT;
BEGIN
	fward:=xaxis;
	rward:=yaxis;
	down:=zaxis;
	XNXLMath.orrot(fward, zaxis, cameraphi);  
	XNXLMath.orrot(rward, zaxis, cameraphi);  
	XNXLMath.orrot(fward, rward, cameratheta);  
	XNXLMath.orrot(down, rward, cameratheta);  
	FWard:=fward;
	w:= aperture.width;
	h:=aperture.height;
	w:=fisheye*w;
	h:= fisheye*h;
	dtheta := w / W;
	dphi := h/ H;
	theta := -w / 2;
	FOR i := 0 TO Base.W - 1  DO
		theta := theta + dtheta;
		phi :=  -aperture.height / 2; 	(*bottom*)
		FOR reversej := 0 TO Base.H - 1 DO
			j:= (Base.H -1)-reversej;
			phi := phi + dphi;
			Base.rays[i, j] := Base.blankray;		
			angletoray(Base.rays[i, j],theta,phi);
			XNXLMath.orrot(Base.rays[i, j].dxyz, zaxis, cameraphi);  	
		   	XNXLMath.orrot(Base.rays[i, j].dxyz, rward, cameratheta);  			  	  								  	  		
			IF Base.rays[i, j].dxyz.x < 0 THEN Base.rays[i, j].di := FALSE  ELSE Base.rays[i, j].di := TRUE END; 
			IF Base.rays[i, j].dxyz.y < 0 THEN Base.rays[i, j].dj := FALSE  ELSE Base.rays[i, j].dj := TRUE END;
			IF Base.rays[i, j].dxyz.z < 0 THEN Base.rays[i, j].dk := FALSE  ELSE Base.rays[i, j].dk := TRUE END;		
			Base.rays[i, j].lxyz := cam;
			Base.rays[i, j].xyz := cam;
			Base.rays[i, j].oxyz:=Base.rays[i, j].xyz;
		END
	END;
	IF ~XNXLRetina.SACCADE THEN
	w:= aperture.width/3;
	h:=aperture.height/3;
	w:=fisheye*w;
	h:= fisheye*h;
	dtheta := w / W;
	dphi := h/ H;
	theta := -w / 2;
	FOR i := 0 TO Base.W - 1  DO
		theta := theta + dtheta;
		phi :=  -aperture.height / 6; 	(*bottom*)
		FOR reversej := 0 TO Base.H - 1 DO
			j:= (Base.H -1)-reversej;
			phi := phi + dphi;
			Base.frays[i, j] := Base.blankray;		
			angletoray(Base.frays[i, j],theta,phi);
			XNXLMath.orrot(Base.frays[i, j].dxyz, zaxis, cameraphi);  	
		   	XNXLMath.orrot(Base.frays[i, j].dxyz, rward, cameratheta);  			  	  								  	  		
			IF Base.frays[i, j].dxyz.x < 0 THEN Base.frays[i, j].di := FALSE  ELSE Base.frays[i, j].di := TRUE END; 
			IF Base.frays[i, j].dxyz.y < 0 THEN Base.frays[i, j].dj := FALSE  ELSE Base.frays[i, j].dj := TRUE END;
			IF Base.frays[i, j].dxyz.z < 0 THEN Base.frays[i, j].dk := FALSE  ELSE Base.frays[i, j].dk := TRUE END;		
			Base.frays[i, j].lxyz := cam;
			Base.frays[i, j].xyz := cam;
			Base.frays[i, j].oxyz:=Base.frays[i, j].dxyz;
		END
	END
	END;
END initrays;

PROCEDURE trace;
VAR
	i, j, ii,jj,q,z: LONGINT;
	ry: Base.Ray;
	r, g, b: LONGINT;
	R,G,B:REAL;
	lr, lg,lb,nlr,nlg,nlb: LONGINT;
	fr,fg,fb: REAL;
BEGIN
	XNXLRetina.go
END trace; 

PROCEDURE left (th: REAL);
BEGIN
	IF ~orientationlock THEN
		cvl:=cvl+th;
		cameratheta:=cameratheta+cvl;
		IF cameratheta> 6.28 THEN cameratheta := cameratheta-6.28 END;
		IF cameratheta< -6.28 THEN cameratheta := cameratheta+6.28 END
	END
END left;

PROCEDURE up (ph: REAL);
BEGIN
	cvu:=cvu+ph;
	cameraphi := cameraphi + cvu;
	IF cameraphi > 1.6 THEN cameraphi := 1.6 END;
	IF cameraphi < -1.6 THEN cameraphi := -1.6 END 	
END up;

PROCEDURE tick;
VAR
	oldcam:PT;
	a,b,c,maxspeed: REAL;
BEGIN
	d2c:=Base.distance(worldcenter,cam);
	move;	
	cam.x := cx; cam.y := cy; cam.z := cz;
	a:=FWard.x*Base.cursordist; b:=FWard.y*Base.cursordist; c:=FWard.z*Base.cursordist;
	Base.updatecamera(cx,cy,cz,a,b,c);	
	IF SCULL THEN GLIDE:=0.7 ELSE GLIDE:=0.85 END;		
	cvz := cvz*GLIDE-Base.gravity;	cvx := cvx*GLIDE; cvy := cvy*GLIDE; 
	cvl :=cvl*0.0; cvu := cvu*0.0;
	initrays;
	IF AVATAR THEN XNXLMarsh.drawavatar END;
	IF MOUSEDOWN THEN
		IF MB=3 THEN 
			XNXLMarsh.drawcursor; 
			XNXLMarsh.draw 
		ELSIF MB=1 THEN XNXLMarsh.draw
		END;
	END;	
	trace;
	Base.tick;
END tick;
 
END Camera;

VAR
	FRAMERATE: REAL;
	rand: Random.Generator;
	wcount: INTEGER;
	frame, tickframe: LONGINT;
	frames: ARRAY 10000 OF SNAP;
	movierecording, movieplaying: BOOLEAN; 
	frtrace, foveate: BOOLEAN;
	foveasize, fovealeft, fovearight, foveadown, foveaup: LONGINT;
	DEATH, wLOOK, MOUSEDOWN, PAUSE: BOOLEAN;
	tracetiled,avtoggle, capturethemouse: BOOLEAN;
	framecount: LONGINT;
	multicore,filtor,uplock,rollcam,suppressz: BOOLEAN;
	camrotinertia:REAL;
	framedt: REAL;
	pkeys: SET;	
	pointerx, pointerdx, pointery, pointerdy: LONGINT;
	R1,R2: REAL;
	xbuttons: SET;
	compstatus: X11.ComposeStatus;
	update: LONGINT;
	MMseen, MRseen: BOOLEAN;	
	noEventCount: LONGINT;
	display : X11.DisplayPtr;
	screen:LONGINT;
	win : X11.Window ;
	glctx : GL.GLXContext;  
	fullscreen, fullwindow, windecorations : BOOLEAN;	
	doubleBuffered: BOOLEAN;   	
	winAttr : Api.XSetWindowAttributes; (* set window attributes*)
	wmDelete: X11.Atom;	
	W,H, width, height, mousex, mousey : LONGINT; (* size of window *)
	desktopMode: V.XF86VidModeModeInfo;
	event: Api.XEvent;  
	keySymbol: ARRAY 256 OF LONGINT;
	currX, currY: LONGINT;
	alive, debug, debugevents : BOOLEAN; (* for main loop control *)	
	drawMode:LONGINT; (* fill, lines, points *)
	texture: ARRAY [3] OF GL.GLuint;  (* Storage For 3 Textures  *)
	LightAmbient, LightDiffuse, LightPosition: ARRAY [4] OF GL.GLfloat;
	LightAmbient2, LightDiffuse2, LightPosition2: ARRAY [4] OF GL.GLfloat;	
	red, green, blue, lightPos: ARRAY [4] OF GL.GLfloat; 
	xrot, yrot, zrot: REAL;  (* X, Y  rotation *)
	axis: CHAR;
	z, zee : REAL; (* depth into the screen *)
	xspeed, yspeed: REAL;
	light, blend,raster: BOOLEAN; (* Whether or not lighting is on,  Whether or not blending is on *)
	filter: LONGINT; (* type of filter *)		
	ptlist: LONGINT;	
	CAPTUREMOUSE, GO,TRAIL,AVATAR,SCULL,ZENITHER,METASLASH: BOOLEAN;	
	camera:Camera;
	fogdensity:REAL;
	leftsticky,upsticky: REAL;
	centerx,centery: LONGINT;
	MB:INTEGER;
	GLIDE:REAL;
	FR: LONGINT;

PROCEDURE angletoray(VAR ray: Base.Ray; theta,phi: REAL);
VAR d: REAL;
BEGIN
	ray.dxyz.x := Math.cos(theta) * Math.cos(phi);
	ray.dxyz.y := Math.sin(theta) * Math.cos(phi);
	ray.dxyz.z := Math.sin(phi);
	d := Math.sqrt(ray.dxyz.x*ray.dxyz.x + ray.dxyz.y* ray.dxyz.y+ray.dxyz.z*ray.dxyz.z);  (* Norma! Liza! Ray! Front and center, oh dark thirty!*)
	ray.dxyz.x := ray.dxyz.x/d;
	ray.dxyz.y := ray.dxyz.y/d;
	ray.dxyz.z := ray.dxyz.z/d; 
END angletoray; 

PROCEDURE carttosph(VAR p: PT; theta, phi: REAL);
BEGIN
	p.x := Math.cos(theta) * Math.cos(phi);
	p.y := Math.sin(theta) * Math.cos(phi);
	p.z := Math.sin(phi);
END carttosph; 

PROCEDURE sphtocart( p: PT; VAR theta, phi: REAL);
VAR
	x,y, z: REAL;
BEGIN
	x := p.x; y := p.y; z := 0;
	Base.normalize(x,y,z);
	theta := XNXLMath.arccos(x);	
	phi := XNXLMath.arccos(1-p.z);
END sphtocart;

PROCEDURE ddray(VAR ray: Base.Ray); 
BEGIN
	ray.ddxyz.x := ray.dxyz.x/10000;
	ray.ddxyz.y := ray.dxyz.y/10000;
	ray.ddxyz.z := ray.dxyz.z/10000; 
END ddray; 

PROCEDURE gray(VAR ray: Ray);
VAR
	gray: REAL;
BEGIN
	gray := (ray.r + ray.g + ray.b)/3;
	ray.r := gray; ray.g := gray; ray.b := gray;
END gray;

PROCEDURE speedup;
BEGIN
	IF Base.speed < 0.001 THEN Base.speed := Base.speed * 1.5 END
END speedup;

PROCEDURE slowdown;
BEGIN
	IF Base.speed > 0.00000000001 THEN Base.speed := Base.speed/1.5 END
END slowdown;

PROCEDURE cursorpush;
BEGIN
	Base.cursordist:=Base.cursordist*1.1
END cursorpush;

PROCEDURE cursorpull;
BEGIN
	Base.cursordist:=Base.cursordist*0.9 
END cursorpull;

PROCEDURE vlnext;
BEGIN
	XNXLMarsh.	incVN;
END vlnext;

PROCEDURE vllast;
BEGIN
	XNXLMarsh.	decVN;
END vllast;

PROCEDURE cursorsizeplus;
BEGIN
	XNXLMarsh.	cursorsize:=ENTIER(XNXLMarsh.cursorsize/1.5)
END cursorsizeplus;

PROCEDURE cursorsizeminus;
BEGIN
	XNXLMarsh.	cursorsize:=ENTIER(XNXLMarsh.cursorsize*1.5)
END cursorsizeminus;

(************** XLTL AMALGAMATION FOLLOWS *)

PROCEDURE Reshape(w, h: LONGINT);
BEGIN	
GL.SetFCR();
	GL.glViewport(0, 0, w, h);
	GL.glClearColor(0.0, 0.0, 0.0, 0.0);
	GL.glMatrixMode(GLC.GL_PROJECTION);
	GL.glLoadIdentity();
	GL.glFrustum(-1,1,-1,1, 0.5, 60); 
	GL.glMatrixMode(GLC.GL_MODELVIEW);
	GL.glLoadIdentity(); 
GL.DelFCR();
	centerx:= w DIV 2; centery:= h DIV 2;
END Reshape;

PROCEDURE GLClose;
VAR res: LONGINT;
BEGIN
	(* do we have a rendering context *)
	IF glctx # 0 THEN
		(* Release the context *)
	    	res := GL.glXMakeCurrent(display, 0, 0);
	    	(* Delete the context *)
		GL.glXDestroyContext(display, glctx);
		glctx := 0;
	END;
	(* switch back to original desktop resolution if we were in fullscreen *)
	IF fullscreen THEN                                                                             
		res := V.XF86VidModeSwitchToMode(display, screen, desktopMode);
		res := V.XF86VidModeSetViewPort(display, screen, 0, 0); 
	END;                 	
	(* do we have a window *)
	IF win # 0 THEN
		(* Unmap the window*)
		Api.UnmapWindow(display, win);
		(* Destroy the window *)
		res:= Api.DestroyWindow(display, win);
		win := 0;
	END;	
	(* do we have a display *)
	IF display # 0 THEN	
		res := Api.CloseDisplay(display);
	END;
END GLClose;

PROCEDURE Close*;
VAR i: INTEGER;
BEGIN
	Base.worldalive:=FALSE;
	GLClose;
END Close;
	
PROCEDURE  InitWindow(w, h: LONGINT; CONST title: ARRAY OF CHAR);
VAR 
	res: LONGINT;
	masks: LONGINT;
	buf: X11.Buffer;
	attrib : ARRAY [*] OF GL.GLint;  (* attributes of GL window *) 
	visinfoptr : Api.VisualInfoPtr;
	gwa : Api.XWindowAttributes; (* get window attributes *) 	
	swa : Api.XSetWindowAttributes; (* set window attributes*)
	cmap : X11.Colormap; (* colormap for window *)	
		
BEGIN
display := X11.OpenDisplay(0);
 IF display =0 THEN
 	Out.String(" cannot connect to X server"); Out.Ln; 
	Close;
     RETURN;
END;  

  NEW(attrib, 13);
  attrib[0] := GLC.GLX_RGBA;
  attrib[1] := GLC.GLX_DOUBLEBUFFER;
  attrib[2] := GLC.GLX_DEPTH_SIZE;	attrib[3] := 24; 
  attrib[4] := GLC.GLX_STENCIL_SIZE;	attrib[5] := 8; 
  attrib[6] := GLC.GLX_RED_SIZE;  	attrib[7] := 8;
  attrib[8] := GLC.GLX_GREEN_SIZE;	attrib[9] := 8;
  attrib[10] := GLC.GLX_RED_SIZE;	attrib[11] := 8;
  attrib[12] := 0 ;

 (*try to find a visual with this attribs *)	
  visinfoptr := GL.glXChooseVisual(display, 0 , SYSTEM.ADR(attrib[0]));

 IF visinfoptr = NIL THEN
  	IF debug THEN Out.String(" NO appropriate visual found"); Out.Ln; END;
  	Close;
     RETURN;
 ELSE 
	 IF debug THEN 
		 Out.String("visinfoptr.depth= "); Out.Int(visinfoptr.depth,0); Out.Ln;
	 	Out.String("visinfoptr.visual ");  Out.Int(visinfoptr.visualID, 0); Out.Ln; 
	END;
END;

 cmap := X11.CreateColormap(display, X11.DefaultRootWindow(display), visinfoptr.visual, X11.AllocNone);
 IF cmap = 0 THEN
 	IF debug THEN 
	 	Out.String(" cannot create colormap"); Out.Ln; 
	 	X11.GetErrorText(display, cmap, buf, LEN(buf));
	 	Out.String("ERROR: CreateColormap = "); Out.String(buf); Out.Ln;
 	END;
 END;

 (* window event masks *)	
masks :=  Api.KeyPressMask + Api.KeyReleaseMask + Api.ButtonPressMask+ Api.ButtonReleaseMask + Api.PointerMotionMask + Api.ButtonMotionMask+ Api.ExposureMask + Api.StructureNotifyMask + Api.FocusChangeMask;

  (* window attributes *)
 swa.backgroundPixel := 0;
 swa.borderPixel := 0;
 swa.colormap := cmap;
 swa.eventMask := masks;
 
 masks :=  Api.CWBackPixel + Api.CWBorderPixel + Api.CWColormap + Api.CWEventMask;

 win := Api.CreateWindow(display, X11.DefaultRootWindow(display), 0, 0, w, h,
		        0, visinfoptr.depth, Api.InputOutput,  visinfoptr.visual, masks, swa);

 (* show window *)	
  Api.MapWindow(display, win);

 (* set title of window *)	 
 res := Api.StoreName(display, win, title); 

(* create GL context *)
 (* GL_TRUE: Use direct rendering, GL_FLSSE: use X server for rendering *)
 glctx := GL.glXCreateContext(display, visinfoptr, 0, GLC.GL_FALSE); 
	IF debug THEN Out.String("glXCreateContext glctx= "); Out.Int(glctx, 0); Out.Ln; END;
 res := GL.glXMakeCurrent(display, win, glctx);
	IF debug THEN  Out.String("glXMakeCurrent res= "); Out.Int(res, 0); Out.Ln; END;
	SuppressX11Cursors(display,win);

END InitWindow;

 PROCEDURE SuppressX11Cursors( disp: X11.DisplayPtr; win: X11.Window);
VAR
	fg : X11.Color;
	pixmap: X11.Pixmap;
	noCursor: X11.Cursor;
	data: ARRAY 8 OF CHAR;
	i : LONGINT;
	BEGIN
	fg.red := 0;  fg.green := 0;  fg.blue :=0;
	FOR i:=0 TO 7 DO data[i] := 0X  END;
	pixmap := X11.CreateBitmapFromData( disp, win, SYSTEM.ADR( data[0] ), 8, 8 );
	noCursor := X11.CreatePixmapCursor( disp, pixmap, pixmap, SYSTEM.ADR( fg ), SYSTEM.ADR( fg ), 0, 0 );
	X11.DefineCursor( disp, win, noCursor ); 
END SuppressX11Cursors;

PROCEDURE ToggleDecorations;
VAR 	hints: Hints; 
	property: X11.Atom;
BEGIN
	IF  ~fullwindow OR  ~fullscreen THEN
		hints.flags := {1};
 		windecorations := ~windecorations;
 		IF windecorations THEN hints.decorations := 0; ELSE hints.decorations := 1; END;
 		X11.Lock;
		property := Api.InternAtom(display, "_MOTIF_WM_HINTS", Api.True); 
		X11.ChangeProperty(display, win, property, property, 32, Api.PropModeReplace, SYSTEM.ADR(hints), 5);
		X11.Unlock;
	END;	
END ToggleDecorations;

PROCEDURE Wr(CONST str: ARRAY OF CHAR);
BEGIN
	IF debugevents THEN Out.String(str); Out.Ln END;
END Wr;

PROCEDURE CheckAlternateKeys( VAR mb: SET );
BEGIN
	IF ~MMseen & (ControlKeyBit IN xbuttons) THEN INCL( mb, MM ) END;
	IF ~MRseen & (Mod1KeyBit IN xbuttons) THEN INCL( mb, MR ) END
END CheckAlternateKeys;


PROCEDURE SendMouseMsg( x, y, dz: LONGINT; xbuttons: SET );
VAR mm: Inputs.AbsMouseMsg;
BEGIN
	mm.keys := {};
	mm.x := x;  mm.y := y;  mm.dz := dz;
	IF Button1Bit IN xbuttons THEN  INCL( mm.keys, ML )  END;
	IF Button2Bit IN xbuttons THEN  INCL( mm.keys, MM );  MMseen := TRUE  END;
	IF Button3Bit IN xbuttons THEN  INCL( mm.keys, MR );  MRseen := TRUE  END;
	IF ~(MMseen & MRseen) THEN  CheckAlternateKeys( mm.keys )  END;
(*	Inputs.mouse.Handle( mm );*)
	mousex := x; mousey:= y;
END SendMouseMsg;

PROCEDURE SendKeyboardMsg( km: Inputs.KeyboardMsg );
BEGIN
     IF km.ch='q' THEN alive := FALSE; END;
END SendKeyboardMsg;

PROCEDURE KeyState( ): SET;
VAR keys: SET;
BEGIN
	keys := {};
	IF ShiftKeyBit IN xbuttons THEN  INCL( keys, Inputs.LeftShift )  END;
	IF ControlKeyBit IN xbuttons THEN  INCL( keys, Inputs.LeftCtrl )  END;
	IF Mod1KeyBit IN xbuttons THEN  INCL( keys, Inputs.LeftAlt )  END;
	IF Mod4KeyBit IN xbuttons THEN  INCL( keys, Inputs.LeftMeta )  END;
	IF Mod5KeyBit IN xbuttons THEN  INCL( keys, Inputs.RightAlt )  END;
	RETURN keys
END KeyState;

PROCEDURE ToggleFullWindow;
VAR 
	cm: Api.XClientMessageEvent;
	xev: Api.XEvent;
	dl: Api.Data40l;
	wmstate, wmfullscreen: X11.Atom;
	res: LONGINT;
BEGIN
	wmstate := Api.InternAtom(display, "_NET_WM_STATE", Api.False);
	wmfullscreen := Api.InternAtom(display, "_NET_WM_STATE_FULLSCREEN", Api.False);
	fullwindow := ~ fullwindow;
	cm.typ := Api.ClientMessage;
	cm.window := win;
	cm.messageType := wmstate;
	cm.format := 32;
	IF fullwindow THEN dl[0] := 1; ELSE dl[0] := 0; END;
	dl[1] := wmfullscreen;
	dl[2] := 0;
	cm.data:=SYSTEM.VAL(Api.Data40, dl);
	xev := SYSTEM.VAL(Api.XEvent, cm);
	res := Api.SendEvent(display, X11.DefaultRootWindow(display), Api.False, Api.SubstructureNotifyMask, SYSTEM.ADR(xev));
END ToggleFullWindow;

PROCEDURE PollXQueue;
VAR   
	 keysym: X11.KeySym;
	 cm: Api.XClientMessageEvent;
	 ke: Api.XKeyEvent;
	 be: Api.XButtonEvent;
	 cn : Api.XConfigureEvent;
	 ee : Api.XExposeEvent;
	 mne: Api.XMotionEvent;
	 datal: Api.Data40l;
	 atomName: Api.PChar;
	 x,y,xr,yr: LONGINT;
	 warpedthemouse: BOOLEAN;
BEGIN
	WHILE Api.Pending(display) > 0 DO 
			warpedthemouse:=FALSE;
			Api.NextEvent( display, event );
			CASE event.typ OF
			| Api.Expose: 
				ee := SYSTEM.VAL(Api.XExposeEvent, event);
			| Api.ConfigureNotify: 
					cn := SYSTEM.VAL(Api.XConfigureEvent, event);
			                    (* call Reshape only if our window-size changed *)
					IF (cn.width # width) OR  (cn.height # height) THEN
							width := cn.width;
							height := cn.height;
							Reshape(width, height);
					END;				
			| Api.ButtonPress: 
						CAPTUREMOUSE:=TRUE;	
						MOUSEDOWN:=TRUE;
						PAUSE:=FALSE;	
						be := SYSTEM.VAL(Api.XButtonEvent, event);
						CASE be.button OF	
							    Api.Button1:  MB:=1; 
							| Api.Button2:    MB:=2; 
							|  Api.Button3:  MB:=3; 
						 	| Api.Button4: MB:=4; speedup;			
							|Api.Button5: MB:=5; slowdown;
						ELSE
						END
		| Api.ButtonRelease: 
				MOUSEDOWN:=FALSE;
				be := SYSTEM.VAL(Api.XButtonReleasedEvent, event);
				CASE be.button OF
					| Api.Button1:  
					| Api.Button2:  
					| Api.Button3:  
					ELSE  (* ignore *)
					END;				
			| Api.MotionNotify:  
			IF ~warpedthemouse OR TRUE THEN (* only once per frame *)
				mne := SYSTEM.VAL(Api.XMotionEvent, event);			
				x := mne.x; y := mne.y;
				xr := x-centerx; yr :=y-centery;
				MouseVent(xr,yr);	
				IF CAPTUREMOUSE & (ABS(xr)+ABS(yr)>5) THEN
					Api.WarpPointer( display, 0,mne.window, 0,0, 1000,1000, centerx,centery);	
				END;
				warpedthemouse:=TRUE
			END
			| Api.KeyPress: 
							ke := SYSTEM.VAL(Api.XKeyEvent, event);								
							keysym := Api.LookupKeysym(ke, 0);
							KeyEvent(keysym)
	
				ELSE		
				END;
		END;
		
END PollXQueue;

PROCEDURE KeyEvent(keysym: X11.KeySym);	 
VAR
	v: Voxel;   
BEGIN
	IF METASLASH THEN 
		XNXLMarsh.key(CHR(keysym))
	ELSE
	CASE keysym OF
		   KS.XK_s1 : camera.backward(0.51)
		| KS.XK_w1 :camera.forward(0.51) 
		| KS.XK_a1 :camera.leftstrafe(0.5)	
		| KS.XK_d1 :camera.rightstrafe(0.5)
		| KS.XK_e1: camera.upstrafe(0.51)
		| KS.XK_q1: camera.downstrafe(0.51)	
		| KS.XK_Escape: IF CAPTUREMOUSE THEN 
							CAPTUREMOUSE := FALSE;
							PAUSE:=TRUE 
						ELSE
							(* alive:=FALSE *)
						END
		|	KS.XK_F1:
			Close;
			fullscreen := ~ fullscreen;
		| KS.XK_Up: Wr("K_Up"); 
		| KS.XK_Down: Wr("K_Down"); 
		| KS.XK_F11: ToggleFullWindow;
		| KS.XK_F12: ToggleDecorations;
		| KS.XK_n1: leftsticky:=leftsticky*1.3; upsticky:=upsticky*1.3;
		| KS.XK_m1: leftsticky:=leftsticky/1.3; upsticky:=upsticky/1.3;
		| KS.XK_t1: camera.glidetoggle 
		| KS.XK_v1:XNXLMarsh.engrain
(*		| KS.XK_c1:XNXLMarsh.sprout *)
		| KS.XK_x1: XNXLMarsh.incVN;
		| KS.XK_z1: XNXLMarsh.decVN;		
		| KS.XK_2 : Base.DTL:= ENTIER(3*Base.DTL/2)	+1
		| KS.XK_3 : Base.DTL:= ENTIER(2*Base.DTL/3)	+1			
		| KS.XK_h1 : camera.hop	
		| KS.XK_bracketright: camera.aperture.width := camera.aperture.width * 101/99; camera.aperture.height := camera.aperture.height *101/99
		| KS.XK_bracketleft: camera.aperture.width := camera.aperture.width * 99/101; camera.aperture.height := camera.aperture.height *99/101
		| KS.XK_period: camera.stop
		| KS.XK_9: IF XNXLMarsh.cursorsize<99999 THEN XNXLMarsh.cursorsize:=XNXLMarsh.cursorsize * 3; Base.cursordist:=Base.cursordist/2 END
		| KS.XK_0: IF XNXLMarsh.cursorsize>9 THEN XNXLMarsh.cursorsize:=XNXLMarsh.cursorsize DIV 3; Base.cursordist:=Base.cursordist*2 END
		| KS.XK_7: Base.cursordist:=Base.cursordist*1.3
		| KS.XK_8: Base.cursordist:=Base.cursordist/1.3
		| KS.XK_5: FR:=FR+10
		| KS.XK_6: FR:=FR-10	
		| KS.XK_Control_L: AVATAR:=~AVATAR	
		| KS.XK_Control_R: XNXLMarsh.SNIPEINTO:=~XNXLMarsh.SNIPEINTO	
		| KS.XK_l1: XNXLMarsh.lines:=~XNXLMarsh.lines
		| KS.XK_b1: ZENITHER:=~ZENITHER
		| KS.XK_slash: METASLASH:=~METASLASH		
	ELSE	
	END
	END	
END KeyEvent;

PROCEDURE MouseVent(x,y: LONGINT);
VAR
	up,left: REAL;
BEGIN
	left:=x*x;
	up:=y*y;
	IF (left+up)>100 THEN XNXLRetina.SACCADE:=TRUE ELSE XNXLRetina.SACCADE:=FALSE END; 
	IF ~SCULL THEN GLIDE:=0.9 ELSE  GLIDE:=0.5 END;
	IF x<0 THEN	
		IF ~SCULL THEN 
			camera.left(-left/(30000*leftsticky)) 
		ELSIF MOUSEDOWN & (MB=1) THEN 
			camera.left(-left/(30000*leftsticky)) 
		ELSE
			camera.forward(left/100000)
		END
	ELSE	
		IF ~SCULL THEN 
			camera.left(left/(30000*leftsticky))
		ELSIF MOUSEDOWN & (MB=1) THEN 
			camera.left(left/(30000*leftsticky)) 
		ELSE
			camera.forward(left/100000)
		END
	END;
	IF y<0 THEN		
		IF ~SCULL THEN 
			camera.up(up/(30000*leftsticky))
		ELSIF MOUSEDOWN & (MB=1) THEN 
				camera.up(up/(30000*leftsticky)) 
		ELSE
				camera.backward(up/100000)
		END
	ELSE
		IF ~SCULL THEN 
			camera.up(-up/(30000*leftsticky)) 
		ELSIF MOUSEDOWN & (MB=1) THEN 
				camera.up(-up/(30000*leftsticky)) 	
		ELSE
			camera.backward(up/100000);
		END
	END
END MouseVent;

PROCEDURE InitGL;
BEGIN

 (*context.MakeCurrent(); *)
GL.SetFCR();


  (* Enable smooth shading *)
  GL.glShadeModel( GLC.GL_SMOOTH );

  (* Set the background black *)
  GL.glClearColor(0,0,1,0);

  
  
  GL.glColorMaterial(GLC.GL_FRONT_AND_BACK, GLC.GL_AMBIENT_AND_DIFFUSE);

  (* Full Brightness, 50% Alpha ( NEW ) *)
  GL.glColor4f( 1.0, 1.0, 1.0, 0.5);
 
GL.DelFCR();
END InitGL;

PROCEDURE draw;
VAR
	i,j,ii,jj,k,i3,j3,ii3,jj3:LONGINT;
	a,w,h: REAL;
BEGIN
		GL.glMatrixMode(GLC.GL_PROJECTION);
		GL.glLoadIdentity();
		GL.glOrtho(0,Base.W*PX, Base.H*PX,0,0,1);
		GL.glMatrixMode(GLC.GL_MODELVIEW);
		GL.glLoadIdentity();	
		GL.glBegin(GLC.GL_QUADS);	
		i3:=0;
		FOR i:= 0 TO Base.W-2 DO 
			j3:=0; 
			FOR j := 0 TO Base.H-2 DO	
				GL.glColor3f(Base.rays[i,j].r, Base.rays[i,j].g, Base.rays[i,j].b); 
				GL.glVertex2f(i3,j3);
			(*	GL.glColor3f(Base.rays[i+1,j].r, Base.rays[i+1,j].g, Base.rays[i+1,j].b); *)
				GL.glVertex2f(i3+PX,j3);
			(*	GL.glColor3f(Base.rays[i+1,j+1].r, Base.rays[i+1,j+1].g, Base.rays[i+1,j+1].b); *)
				GL.glVertex2f(i3+PX,j3+PX);
			(*	GL.glColor3f(Base.rays[i,j+1].r, Base.rays[i,j+1].g, Base.rays[i,j+1].b); *)
				GL.glVertex2f(i3,j3+PX);	
				j3:=j3+PX
			END;
			i3:=i3+PX
		END;
		IF ~(XNXLRetina.SACCADE) THEN 
			i3:=Base.W;
			FOR i:= 0 TO Base.W-1 DO 
				j3:=Base.H;
				FOR j := 0 TO Base.H-1 DO	
					GL.glColor3f(Base.frays[i,j].r, Base.frays[i,j].g, Base.frays[i,j].b); 
					GL.glVertex2f(i3,j3);
					GL.glVertex2f(i3+1,j3);
					GL.glVertex2f(i3+1,j3+1);
					GL.glVertex2f(i3,j3+1);		
					j3:=j3+1
				END;
				i3:=i3+1 
			END;
		END; 
        GL.glColor3f(Base.rand.Uniform(),Base.rand.Uniform(),Base.rand.Uniform());
        w:=PX*Base.W/2; h:=PX*Base.H/2;
        GL.glVertex2f(w-1,h-1); 
        GL.glVertex2f(w-1,h+1); 
        GL.glVertex2f(w+1,h+1); 
        GL.glVertex2f(w+1,h-1); 
	 GL.glEnd;	
	 GL.glXSwapBuffers(display, win);
END draw;



PROCEDURE mainloop;
VAR
	timer: Kernel.Timer;
	ftimer: Kernel.MilliTimer;
	framems: LONGINT;
BEGIN
	alive:=TRUE;
	NEW(timer);
	NEW(rand);
	WHILE alive DO
		Base.rnf:=rand.Uniform();
		Kernel.SetTimer(ftimer,1000);
		PollXQueue; 
		IF ~PAUSE THEN 
			camera.tick ;
			draw  
		 END;
		framems:=Kernel.Elapsed(ftimer);
		Out.Int(framems, 6);
		Out.Ln;
		IF framems<40 THEN timer.Sleep(40-framems) END;		
		PollXQueue; 
	END;
END mainloop;	

PROCEDURE Open* ;
BEGIN
	Out.Char('*');	
	InitWindow(PX*Base.W, PX*Base.H,'XENOXOLOTL' );	
	InitGL; 	
	XNXLMarsh.dawn;	
	alive:=TRUE;
	NEW(rand);
	foveasize := 100;
	multicore:=TRUE;
	camrotinertia:=100;
	Base.DTL:=150;
	Base.DTL2:=10;
	Base.gravity:=0.0;
	Base.cursordist:=0.01;
	R1:=100;
	R2:=150;
	avtoggle:=TRUE;
	Base.worldalive := TRUE;
	debugevents:=FALSE;
	mousex:=200;
	mousey:=200;
	centerx:=200;
	centery:=200;
	fogdensity:=0.11;
	leftsticky:=50.0;
	upsticky:=50.0;	
	NEW(camera,Base.W, Base.H); 
	mainloop; 
	Out.Char('*');
	FR:=1000;
END Open;

BEGIN

END XNXLRender.Open


SystemTools.FreeDownTo  XNXL~
