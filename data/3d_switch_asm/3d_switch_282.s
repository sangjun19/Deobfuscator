	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-1-2\0"
	.align	3
lC2:
	.ascii "1-2-1\0"
	.align	3
lC3:
	.ascii "1-2-2\0"
	.align	3
lC4:
	.ascii "1-3-1\0"
	.align	3
lC5:
	.ascii "1-3-2\0"
	.align	3
lC6:
	.ascii "1-4-1\0"
	.align	3
lC7:
	.ascii "1-4-2\0"
	.align	3
lC8:
	.ascii "1-5-1\0"
	.align	3
lC9:
	.ascii "1-5-2\0"
	.align	3
lC10:
	.ascii "1-6-1\0"
	.align	3
lC11:
	.ascii "1-6-2\0"
	.align	3
lC12:
	.ascii "1-7-1\0"
	.align	3
lC13:
	.ascii "1-7-2\0"
	.align	3
lC14:
	.ascii "1-8-1\0"
	.align	3
lC15:
	.ascii "1-8-2\0"
	.align	3
lC16:
	.ascii "1-9-1\0"
	.align	3
lC17:
	.ascii "1-9-2\0"
	.align	3
lC18:
	.ascii "2-1-1\0"
	.align	3
lC19:
	.ascii "2-1-2\0"
	.align	3
lC20:
	.ascii "2-2-1\0"
	.align	3
lC21:
	.ascii "2-2-2\0"
	.align	3
lC22:
	.ascii "2-3-1\0"
	.align	3
lC23:
	.ascii "2-3-2\0"
	.align	3
lC24:
	.ascii "2-4-1\0"
	.align	3
lC25:
	.ascii "2-4-2\0"
	.align	3
lC26:
	.ascii "2-5-1\0"
	.align	3
lC27:
	.ascii "2-5-2\0"
	.align	3
lC28:
	.ascii "2-6-1\0"
	.align	3
lC29:
	.ascii "2-6-2\0"
	.align	3
lC30:
	.ascii "2-7-1\0"
	.align	3
lC31:
	.ascii "2-7-2\0"
	.align	3
lC32:
	.ascii "2-8-1\0"
	.align	3
lC33:
	.ascii "2-8-2\0"
	.align	3
lC34:
	.ascii "2-9-1\0"
	.align	3
lC35:
	.ascii "2-9-2\0"
	.align	3
lC36:
	.ascii "3-1-1\0"
	.align	3
lC37:
	.ascii "3-1-2\0"
	.align	3
lC38:
	.ascii "3-2-1\0"
	.align	3
lC39:
	.ascii "3-2-2\0"
	.align	3
lC40:
	.ascii "3-3-1\0"
	.align	3
lC41:
	.ascii "3-3-2\0"
	.align	3
lC42:
	.ascii "3-4-1\0"
	.align	3
lC43:
	.ascii "3-4-2\0"
	.align	3
lC44:
	.ascii "3-5-1\0"
	.align	3
lC45:
	.ascii "3-5-2\0"
	.align	3
lC46:
	.ascii "3-6-1\0"
	.align	3
lC47:
	.ascii "3-6-2\0"
	.align	3
lC48:
	.ascii "3-7-1\0"
	.align	3
lC49:
	.ascii "3-7-2\0"
	.align	3
lC50:
	.ascii "3-8-1\0"
	.align	3
lC51:
	.ascii "3-8-2\0"
	.align	3
lC52:
	.ascii "3-9-1\0"
	.align	3
lC53:
	.ascii "3-9-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 3
	str	w0, [x29, 28]
	mov	w0, 9
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L5
	b	L3
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L6
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L15
	b	L118
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L7
L16:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L18:
	b	L7
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L7
L19:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L21
L20:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L21:
	b	L7
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L23
	b	L7
L22:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L24
L23:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L24:
	b	L7
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L7
L25:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L27
L26:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L27:
	b	L7
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L29
	b	L7
L28:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L30:
	b	L7
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L7
L31:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L33
L32:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L33:
	b	L7
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L35
	b	L7
L34:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L36
L35:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L36:
	b	L7
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L38
	b	L7
L37:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L39
L38:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L39:
	b	L7
L6:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L41
	b	L119
L40:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L42
L41:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L42:
L119:
	nop
L7:
	b	L118
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L46
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L47
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L48
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L49
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L50
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L52
	b	L120
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L44
L53:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L55
L54:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L55:
	b	L44
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L57
	b	L44
L56:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L58
L57:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L58:
	b	L44
L50:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L44
L59:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L61
L60:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L61:
	b	L44
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L63
	b	L44
L62:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L64
L63:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L64:
	b	L44
L48:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L66
	b	L44
L65:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L67
L66:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L67:
	b	L44
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L69
	b	L44
L68:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L70
L69:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L70:
	b	L44
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L44
L71:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L73
L72:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L73:
	b	L44
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L74
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L75
	b	L44
L74:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L76
L75:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L76:
	b	L44
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L77
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L78
	b	L121
L77:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L79
L78:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L79:
L121:
	nop
L44:
	b	L120
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L80
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L82
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L83
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L84
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L85
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L86
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L87
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L89
	b	L122
L88:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L90
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L91
	b	L81
L90:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L92
L91:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L92:
	b	L81
L89:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L94
	b	L81
L93:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L95
L94:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L95:
	b	L81
L87:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L96
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L97
	b	L81
L96:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L98
L97:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L98:
	b	L81
L86:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L99
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L100
	b	L81
L99:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L101
L100:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L101:
	b	L81
L85:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L102
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L103
	b	L81
L102:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L104
L103:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L104:
	b	L81
L84:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L105
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L106
	b	L81
L105:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L107
L106:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L107:
	b	L81
L83:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L108
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L109
	b	L81
L108:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L110
L109:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L110:
	b	L81
L82:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L111
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L112
	b	L81
L111:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L113
L112:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L113:
	b	L81
L80:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L114
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L115
	b	L123
L114:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L116
L115:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L116:
L123:
	nop
L81:
	b	L122
L118:
	nop
	b	L3
L120:
	nop
	b	L3
L122:
	nop
L3:
	mov	w0, 0
	ldp	x29, x30, [sp], 32
LCFI2:
	ret
LFE1:
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x3
	.ascii "zR\0"
	.uleb128 0x1
	.sleb128 -8
	.uleb128 0x1e
	.uleb128 0x1
	.byte	0x10
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB1-.
	.set L$set$2,LFE1-LFB1
	.quad L$set$2
	.uleb128 0
	.byte	0x4
	.set L$set$3,LCFI0-LFB1
	.long L$set$3
	.byte	0xe
	.uleb128 0x20
	.byte	0x9d
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x3
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xd
	.uleb128 0x1d
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xde
	.byte	0xdd
	.byte	0xc
	.uleb128 0x1f
	.uleb128 0
	.align	3
LEFDE1:
	.ident	"GCC: (Homebrew GCC 14.2.0_1) 14.2.0"
	.subsections_via_symbols
