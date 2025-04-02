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
	.ascii "1-1-3\0"
	.align	3
lC3:
	.ascii "1-2-1\0"
	.align	3
lC4:
	.ascii "1-2-2\0"
	.align	3
lC5:
	.ascii "1-2-3\0"
	.align	3
lC6:
	.ascii "2-1-1\0"
	.align	3
lC7:
	.ascii "2-1-2\0"
	.align	3
lC8:
	.ascii "2-1-3\0"
	.align	3
lC9:
	.ascii "2-2-1\0"
	.align	3
lC10:
	.ascii "2-2-2\0"
	.align	3
lC11:
	.ascii "2-2-3\0"
	.align	3
lC12:
	.ascii "3-1-1\0"
	.align	3
lC13:
	.ascii "3-1-2\0"
	.align	3
lC14:
	.ascii "3-1-3\0"
	.align	3
lC15:
	.ascii "3-2-1\0"
	.align	3
lC16:
	.ascii "3-2-2\0"
	.align	3
lC17:
	.ascii "3-2-3\0"
	.align	3
lC18:
	.ascii "4-1-1\0"
	.align	3
lC19:
	.ascii "4-1-2\0"
	.align	3
lC20:
	.ascii "4-1-3\0"
	.align	3
lC21:
	.ascii "4-2-1\0"
	.align	3
lC22:
	.ascii "4-2-2\0"
	.align	3
lC23:
	.ascii "4-2-3\0"
	.align	3
lC24:
	.ascii "5-1-1\0"
	.align	3
lC25:
	.ascii "5-1-2\0"
	.align	3
lC26:
	.ascii "5-1-3\0"
	.align	3
lC27:
	.ascii "5-2-1\0"
	.align	3
lC28:
	.ascii "5-2-2\0"
	.align	3
lC29:
	.ascii "5-2-3\0"
	.align	3
lC30:
	.ascii "6-1-1\0"
	.align	3
lC31:
	.ascii "6-1-2\0"
	.align	3
lC32:
	.ascii "6-1-3\0"
	.align	3
lC33:
	.ascii "6-2-1\0"
	.align	3
lC34:
	.ascii "6-2-2\0"
	.align	3
lC35:
	.ascii "6-2-3\0"
	.align	3
lC36:
	.ascii "7-1-1\0"
	.align	3
lC37:
	.ascii "7-1-2\0"
	.align	3
lC38:
	.ascii "7-1-3\0"
	.align	3
lC39:
	.ascii "7-2-1\0"
	.align	3
lC40:
	.ascii "7-2-2\0"
	.align	3
lC41:
	.ascii "7-2-3\0"
	.align	3
lC42:
	.ascii "8-1-1\0"
	.align	3
lC43:
	.ascii "8-1-2\0"
	.align	3
lC44:
	.ascii "8-1-3\0"
	.align	3
lC45:
	.ascii "8-2-1\0"
	.align	3
lC46:
	.ascii "8-2-2\0"
	.align	3
lC47:
	.ascii "8-2-3\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 8
	str	w0, [x29, 28]
	mov	w0, 2
	str	w0, [x29, 24]
	mov	w0, 3
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L10
	b	L3
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L12
	b	L3
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L100
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L100
L16:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L15
L17:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L15
L14:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
L15:
	b	L100
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L101
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L21
	b	L101
L20:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L19
L21:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L19
L18:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L19:
	b	L101
L100:
	nop
	b	L3
L101:
	nop
	b	L3
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L23
	b	L3
L22:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L102
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L28
	b	L102
L27:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L26
L28:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L26
L25:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L26:
	b	L102
L23:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L103
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L103
L31:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L30
L32:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L30:
	b	L103
L102:
	nop
	b	L3
L103:
	nop
	b	L3
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L34
	b	L3
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L104
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L39
	b	L104
L38:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L37
L39:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L37
L36:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L37:
	b	L104
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L105
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L43
	b	L105
L42:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L41
L43:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L41
L40:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L41:
	b	L105
L104:
	nop
	b	L3
L105:
	nop
	b	L3
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L45
	b	L3
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L106
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L106
L49:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L48
L50:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L48:
	b	L106
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L107
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L107
L53:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L52
L54:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L52
L51:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L52:
	b	L107
L106:
	nop
	b	L3
L107:
	nop
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L55
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L56
	b	L3
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L108
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L60
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L61
	b	L108
L60:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L59
L61:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L59
L58:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
L59:
	b	L108
L56:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L109
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L64
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L65
	b	L109
L64:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L63
L65:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L63
L62:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L63:
	b	L109
L108:
	nop
	b	L3
L109:
	nop
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L66
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L67
	b	L3
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L69
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L110
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L110
L71:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L70
L72:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L70
L69:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
L70:
	b	L110
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L111
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L76
	b	L111
L75:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L74
L76:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L74
L73:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L74:
	b	L111
L110:
	nop
	b	L3
L111:
	nop
	b	L3
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L78
	b	L3
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L80
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L112
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L83
	b	L112
L82:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L81
L83:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L81
L80:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
L81:
	b	L112
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L113
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L86
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L87
	b	L113
L86:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	b	L85
L87:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L85
L84:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L85:
	b	L113
L112:
	nop
	b	L3
L113:
	nop
	b	L3
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L89
	b	L116
L88:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L91
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L114
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L93
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L94
	b	L114
L93:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L92
L94:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	b	L92
L91:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
L92:
	b	L114
L89:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L115
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L98
	b	L115
L97:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L96
L98:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L96
L95:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L96:
	b	L115
L114:
	nop
	b	L116
L115:
	nop
L116:
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
