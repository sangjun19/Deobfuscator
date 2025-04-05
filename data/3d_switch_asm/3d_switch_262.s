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
	.ascii "2-1-1\0"
	.align	3
lC15:
	.ascii "2-1-2\0"
	.align	3
lC16:
	.ascii "2-2-1\0"
	.align	3
lC17:
	.ascii "2-2-2\0"
	.align	3
lC18:
	.ascii "2-3-1\0"
	.align	3
lC19:
	.ascii "2-3-2\0"
	.align	3
lC20:
	.ascii "2-4-1\0"
	.align	3
lC21:
	.ascii "2-4-2\0"
	.align	3
lC22:
	.ascii "2-5-1\0"
	.align	3
lC23:
	.ascii "2-5-2\0"
	.align	3
lC24:
	.ascii "2-6-1\0"
	.align	3
lC25:
	.ascii "2-6-2\0"
	.align	3
lC26:
	.ascii "2-7-1\0"
	.align	3
lC27:
	.ascii "2-7-2\0"
	.align	3
lC28:
	.ascii "3-1-1\0"
	.align	3
lC29:
	.ascii "3-1-2\0"
	.align	3
lC30:
	.ascii "3-2-1\0"
	.align	3
lC31:
	.ascii "3-2-2\0"
	.align	3
lC32:
	.ascii "3-3-1\0"
	.align	3
lC33:
	.ascii "3-3-2\0"
	.align	3
lC34:
	.ascii "3-4-1\0"
	.align	3
lC35:
	.ascii "3-4-2\0"
	.align	3
lC36:
	.ascii "3-5-1\0"
	.align	3
lC37:
	.ascii "3-5-2\0"
	.align	3
lC38:
	.ascii "3-6-1\0"
	.align	3
lC39:
	.ascii "3-6-2\0"
	.align	3
lC40:
	.ascii "3-7-1\0"
	.align	3
lC41:
	.ascii "3-7-2\0"
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
	mov	w0, 7
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
	cmp	w0, 7
	beq	L6
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L94
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L13
	b	L94
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L15
	b	L7
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
	b	L7
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L18
	b	L7
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
	b	L7
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L21
	b	L7
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
	b	L7
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L7
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
	b	L7
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L27
	b	L7
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
	b	L7
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L30
	b	L7
L29:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L31
L30:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L31:
	b	L7
L6:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L33
	b	L95
L32:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L34
L33:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L34:
L95:
	nop
L7:
	b	L94
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L35
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L96
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L37
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L96
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L38
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L96
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L96
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L96
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L42
	b	L96
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L44
	b	L36
L43:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L45
L44:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L45:
	b	L36
L42:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L47
	b	L36
L46:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L48:
	b	L36
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L36
L49:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L51
L50:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L51:
	b	L36
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L52
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L53
	b	L36
L52:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L54
L53:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L54:
	b	L36
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L56
	b	L36
L55:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L57
L56:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L57:
	b	L36
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L59
	b	L36
L58:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L60
L59:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L60:
	b	L36
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L62
	b	L97
L61:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L63
L62:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L63:
L97:
	nop
L36:
	b	L96
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L64
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L98
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L66
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L98
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L67
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L98
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L68
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L98
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L69
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L98
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L71
	b	L98
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L72
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L73
	b	L65
L72:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L74
L73:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L74:
	b	L65
L71:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L76
	b	L65
L75:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L77
L76:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L77:
	b	L65
L69:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L78
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L79
	b	L65
L78:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L80
L79:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L80:
	b	L65
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L81
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L82
	b	L65
L81:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L83
L82:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L83:
	b	L65
L67:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L85
	b	L65
L84:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L86
L85:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L86:
	b	L65
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L88
	b	L65
L87:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L89
L88:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L89:
	b	L65
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L90
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L91
	b	L99
L90:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L92
L91:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L92:
L99:
	nop
L65:
	b	L98
L94:
	nop
	b	L3
L96:
	nop
	b	L3
L98:
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
