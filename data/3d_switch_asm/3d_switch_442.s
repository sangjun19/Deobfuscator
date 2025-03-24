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
	.ascii "2-1-1\0"
	.align	3
lC11:
	.ascii "2-1-2\0"
	.align	3
lC12:
	.ascii "2-2-1\0"
	.align	3
lC13:
	.ascii "2-2-2\0"
	.align	3
lC14:
	.ascii "2-3-1\0"
	.align	3
lC15:
	.ascii "2-3-2\0"
	.align	3
lC16:
	.ascii "2-4-1\0"
	.align	3
lC17:
	.ascii "2-4-2\0"
	.align	3
lC18:
	.ascii "2-5-1\0"
	.align	3
lC19:
	.ascii "2-5-2\0"
	.align	3
lC20:
	.ascii "3-1-1\0"
	.align	3
lC21:
	.ascii "3-1-2\0"
	.align	3
lC22:
	.ascii "3-2-1\0"
	.align	3
lC23:
	.ascii "3-2-2\0"
	.align	3
lC24:
	.ascii "3-3-1\0"
	.align	3
lC25:
	.ascii "3-3-2\0"
	.align	3
lC26:
	.ascii "3-4-1\0"
	.align	3
lC27:
	.ascii "3-4-2\0"
	.align	3
lC28:
	.ascii "3-5-1\0"
	.align	3
lC29:
	.ascii "3-5-2\0"
	.align	3
lC30:
	.ascii "4-1-1\0"
	.align	3
lC31:
	.ascii "4-1-2\0"
	.align	3
lC32:
	.ascii "4-2-1\0"
	.align	3
lC33:
	.ascii "4-2-2\0"
	.align	3
lC34:
	.ascii "4-3-1\0"
	.align	3
lC35:
	.ascii "4-3-2\0"
	.align	3
lC36:
	.ascii "4-4-1\0"
	.align	3
lC37:
	.ascii "4-4-2\0"
	.align	3
lC38:
	.ascii "4-5-1\0"
	.align	3
lC39:
	.ascii "4-5-2\0"
	.align	3
lC40:
	.ascii "5-1-1\0"
	.align	3
lC41:
	.ascii "5-1-2\0"
	.align	3
lC42:
	.ascii "5-2-1\0"
	.align	3
lC43:
	.ascii "5-2-2\0"
	.align	3
lC44:
	.ascii "5-3-1\0"
	.align	3
lC45:
	.ascii "5-3-2\0"
	.align	3
lC46:
	.ascii "5-4-1\0"
	.align	3
lC47:
	.ascii "5-4-2\0"
	.align	3
lC48:
	.ascii "5-5-1\0"
	.align	3
lC49:
	.ascii "5-5-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 5
	str	w0, [x29, 28]
	mov	w0, 5
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L7
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L114
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L13
	b	L114
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L15
	b	L9
L14:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L16
L15:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L16:
	b	L9
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L18
	b	L9
L17:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L19
L18:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L19:
	b	L9
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L21
	b	L9
L20:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L22:
	b	L9
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L9
L23:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L25
L24:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L25:
	b	L9
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L27
	b	L115
L26:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L28
L27:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L28:
L115:
	nop
L9:
	b	L114
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L34
	b	L116
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L36
	b	L30
L35:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L37
L36:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L37:
	b	L30
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L39
	b	L30
L38:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L40
L39:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L40:
	b	L30
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L42
	b	L30
L41:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L43
L42:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L43:
	b	L30
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L45
	b	L30
L44:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L46
L45:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L46:
	b	L30
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L48
	b	L117
L47:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L49
L48:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L49:
L117:
	nop
L30:
	b	L116
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L50
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L118
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L55
	b	L118
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L57
	b	L51
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
	b	L51
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L51
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
	b	L51
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L63
	b	L51
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
	b	L51
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L66
	b	L51
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
	b	L51
L50:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L69
	b	L119
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
L119:
	nop
L51:
	b	L118
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L71
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L73
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L74
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L120
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L76
	b	L120
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L77
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L78
	b	L72
L77:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L79
L78:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L79:
	b	L72
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L80
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L81
	b	L72
L80:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L82
L81:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L82:
	b	L72
L74:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L84
	b	L72
L83:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L85
L84:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L85:
	b	L72
L73:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L86
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L87
	b	L72
L86:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L88
L87:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L88:
	b	L72
L71:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L89
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L90
	b	L121
L89:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L91
L90:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L91:
L121:
	nop
L72:
	b	L120
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L92
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L94
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L95
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L122
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L96
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L97
	b	L122
L96:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L98
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L99
	b	L93
L98:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L100
L99:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L100:
	b	L93
L97:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L102
	b	L93
L101:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L103
L102:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L103:
	b	L93
L95:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L104
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L105
	b	L93
L104:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L106
L105:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L106:
	b	L93
L94:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L107
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L108
	b	L93
L107:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L109
L108:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L109:
	b	L93
L92:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L110
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L111
	b	L123
L110:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L112
L111:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L112:
L123:
	nop
L93:
	b	L122
L114:
	nop
	b	L3
L116:
	nop
	b	L3
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
