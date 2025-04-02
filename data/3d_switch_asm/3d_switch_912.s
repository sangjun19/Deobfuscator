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
	.ascii "2-1-1\0"
	.align	3
lC5:
	.ascii "2-1-2\0"
	.align	3
lC6:
	.ascii "2-2-1\0"
	.align	3
lC7:
	.ascii "2-2-2\0"
	.align	3
lC8:
	.ascii "3-1-1\0"
	.align	3
lC9:
	.ascii "3-1-2\0"
	.align	3
lC10:
	.ascii "3-2-1\0"
	.align	3
lC11:
	.ascii "3-2-2\0"
	.align	3
lC12:
	.ascii "4-1-1\0"
	.align	3
lC13:
	.ascii "4-1-2\0"
	.align	3
lC14:
	.ascii "4-2-1\0"
	.align	3
lC15:
	.ascii "4-2-2\0"
	.align	3
lC16:
	.ascii "5-1-1\0"
	.align	3
lC17:
	.ascii "5-1-2\0"
	.align	3
lC18:
	.ascii "5-2-1\0"
	.align	3
lC19:
	.ascii "5-2-2\0"
	.align	3
lC20:
	.ascii "6-1-1\0"
	.align	3
lC21:
	.ascii "6-1-2\0"
	.align	3
lC22:
	.ascii "6-2-1\0"
	.align	3
lC23:
	.ascii "6-2-2\0"
	.align	3
lC24:
	.ascii "7-1-1\0"
	.align	3
lC25:
	.ascii "7-1-2\0"
	.align	3
lC26:
	.ascii "7-2-1\0"
	.align	3
lC27:
	.ascii "7-2-2\0"
	.align	3
lC28:
	.ascii "8-1-1\0"
	.align	3
lC29:
	.ascii "8-1-2\0"
	.align	3
lC30:
	.ascii "8-2-1\0"
	.align	3
lC31:
	.ascii "8-2-2\0"
	.align	3
lC32:
	.ascii "9-1-1\0"
	.align	3
lC33:
	.ascii "9-1-2\0"
	.align	3
lC34:
	.ascii "9-2-1\0"
	.align	3
lC35:
	.ascii "9-2-2\0"
	.align	3
lC36:
	.ascii "10-1-1\0"
	.align	3
lC37:
	.ascii "10-1-2\0"
	.align	3
lC38:
	.ascii "10-2-1\0"
	.align	3
lC39:
	.ascii "10-2-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 10
	str	w0, [x29, 28]
	mov	w0, 2
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 10
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 10
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 9
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 9
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L10
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L12
	b	L3
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L3
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L15
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
	b	L15
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L104
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
L104:
	nop
L15:
	b	L3
L12:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L23
	b	L3
L22:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L24
L25:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L27
L26:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L27:
	b	L24
L23:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L29
	b	L105
L28:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L30:
L105:
	nop
L24:
	b	L3
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L32
	b	L3
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L35
	b	L33
L34:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L36
L35:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L36:
	b	L33
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L38
	b	L106
L37:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L39
L38:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L39:
L106:
	nop
L33:
	b	L3
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L41
	b	L3
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L44
	b	L42
L43:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L45
L44:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L45:
	b	L42
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L47
	b	L107
L46:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L48:
L107:
	nop
L42:
	b	L3
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L50
	b	L3
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L52
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L53
	b	L51
L52:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L54
L53:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L54:
	b	L51
L50:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L56
	b	L108
L55:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L57
L56:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L57:
L108:
	nop
L51:
	b	L3
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L58
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L59
	b	L3
L58:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L62
	b	L60
L61:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L63
L62:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L63:
	b	L60
L59:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L64
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L65
	b	L109
L64:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L66
L65:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L66:
L109:
	nop
L60:
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L67
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L68
	b	L3
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L70
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L71
	b	L69
L70:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L72
L71:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L72:
	b	L69
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L73
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L74
	b	L110
L73:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L75
L74:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L75:
L110:
	nop
L69:
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L76
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L77
	b	L3
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L80
	b	L78
L79:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L81
L80:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L81:
	b	L78
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L83
	b	L111
L82:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L84
L83:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L84:
L111:
	nop
L78:
	b	L3
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L85
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L86
	b	L3
L85:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L88
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L89
	b	L87
L88:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L90
L89:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L90:
	b	L87
L86:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L91
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L92
	b	L112
L91:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L93
L92:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L93:
L112:
	nop
L87:
	b	L3
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L94
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L95
	b	L114
L94:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L98
	b	L96
L97:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L99
L98:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L99:
	b	L96
L95:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L100
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L101
	b	L113
L100:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L102
L101:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L102:
L113:
	nop
L96:
L114:
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
