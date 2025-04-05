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
	.ascii "1-1-4\0"
	.align	3
lC4:
	.ascii "1-2-1\0"
	.align	3
lC5:
	.ascii "1-2-2\0"
	.align	3
lC6:
	.ascii "1-2-3\0"
	.align	3
lC7:
	.ascii "1-2-4\0"
	.align	3
lC8:
	.ascii "2-1-1\0"
	.align	3
lC9:
	.ascii "2-1-2\0"
	.align	3
lC10:
	.ascii "2-1-3\0"
	.align	3
lC11:
	.ascii "2-1-4\0"
	.align	3
lC12:
	.ascii "2-2-1\0"
	.align	3
lC13:
	.ascii "2-2-2\0"
	.align	3
lC14:
	.ascii "2-2-3\0"
	.align	3
lC15:
	.ascii "2-2-4\0"
	.align	3
lC16:
	.ascii "3-1-1\0"
	.align	3
lC17:
	.ascii "3-1-2\0"
	.align	3
lC18:
	.ascii "3-1-3\0"
	.align	3
lC19:
	.ascii "3-1-4\0"
	.align	3
lC20:
	.ascii "3-2-1\0"
	.align	3
lC21:
	.ascii "3-2-2\0"
	.align	3
lC22:
	.ascii "3-2-3\0"
	.align	3
lC23:
	.ascii "3-2-4\0"
	.align	3
lC24:
	.ascii "4-1-1\0"
	.align	3
lC25:
	.ascii "4-1-2\0"
	.align	3
lC26:
	.ascii "4-1-3\0"
	.align	3
lC27:
	.ascii "4-1-4\0"
	.align	3
lC28:
	.ascii "4-2-1\0"
	.align	3
lC29:
	.ascii "4-2-2\0"
	.align	3
lC30:
	.ascii "4-2-3\0"
	.align	3
lC31:
	.ascii "4-2-4\0"
	.align	3
lC32:
	.ascii "5-1-1\0"
	.align	3
lC33:
	.ascii "5-1-2\0"
	.align	3
lC34:
	.ascii "5-1-3\0"
	.align	3
lC35:
	.ascii "5-1-4\0"
	.align	3
lC36:
	.ascii "5-2-1\0"
	.align	3
lC37:
	.ascii "5-2-2\0"
	.align	3
lC38:
	.ascii "5-2-3\0"
	.align	3
lC39:
	.ascii "5-2-4\0"
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
	mov	w0, 2
	str	w0, [x29, 24]
	mov	w0, 4
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
	cmp	w0, 1
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L9
	b	L3
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L74
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L74
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L15
	b	L74
L14:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L12
L15:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L12
L13:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L12
L11:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L12:
	b	L74
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L75
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L75
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L75
L19:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L17
L20:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L17
L18:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L17
L16:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L17:
	b	L75
L74:
	nop
	b	L3
L75:
	nop
	b	L3
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L22
	b	L3
L21:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L76
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L76
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L28
	b	L76
L27:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L25
L28:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L25
L26:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L25
L24:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L25:
	b	L76
L22:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L77
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L77
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L33
	b	L77
L32:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L30
L33:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L30
L31:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L30:
	b	L77
L76:
	nop
	b	L3
L77:
	nop
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L35
	b	L3
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L78
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L78
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L41
	b	L78
L40:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L38
L41:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L38
L39:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L38
L37:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L38:
	b	L78
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L79
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L79
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L46
	b	L79
L45:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L43
L46:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L43
L44:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L43
L42:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L43:
	b	L79
L78:
	nop
	b	L3
L79:
	nop
	b	L3
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L47
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L48
	b	L3
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L80
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L52
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L80
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L80
L53:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L51
L54:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L51
L52:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L51
L50:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L51:
	b	L80
L48:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L55
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L81
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L81
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L59
	b	L81
L58:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L56
L59:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	b	L56
L57:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L56
L55:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L56:
	b	L81
L80:
	nop
	b	L3
L81:
	nop
	b	L3
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L60
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L61
	b	L84
L60:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L82
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L82
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L66
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L67
	b	L82
L66:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L64
L67:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L64
L65:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L64
L63:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L64:
	b	L82
L61:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L83
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L70
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L83
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L83
L71:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L69
L72:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L69
L70:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L69
L68:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L69:
	b	L83
L82:
	nop
	b	L84
L83:
	nop
L84:
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
