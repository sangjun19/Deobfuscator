	.arch armv8.4-a+fp16+sb+ssbs
	.build_version macos,  15, 0
	.text
	.cstring
	.align	3
lC0:
	.ascii "1-1-1\0"
	.align	3
lC1:
	.ascii "1-2-1\0"
	.align	3
lC2:
	.ascii "1-3-1\0"
	.align	3
lC3:
	.ascii "1-4-1\0"
	.align	3
lC4:
	.ascii "1-5-1\0"
	.align	3
lC5:
	.ascii "1-6-1\0"
	.align	3
lC6:
	.ascii "1-7-1\0"
	.align	3
lC7:
	.ascii "1-8-1\0"
	.align	3
lC8:
	.ascii "2-1-1\0"
	.align	3
lC9:
	.ascii "2-2-1\0"
	.align	3
lC10:
	.ascii "2-3-1\0"
	.align	3
lC11:
	.ascii "2-4-1\0"
	.align	3
lC12:
	.ascii "2-5-1\0"
	.align	3
lC13:
	.ascii "2-6-1\0"
	.align	3
lC14:
	.ascii "2-7-1\0"
	.align	3
lC15:
	.ascii "2-8-1\0"
	.align	3
lC16:
	.ascii "3-1-1\0"
	.align	3
lC17:
	.ascii "3-2-1\0"
	.align	3
lC18:
	.ascii "3-3-1\0"
	.align	3
lC19:
	.ascii "3-4-1\0"
	.align	3
lC20:
	.ascii "3-5-1\0"
	.align	3
lC21:
	.ascii "3-6-1\0"
	.align	3
lC22:
	.ascii "3-7-1\0"
	.align	3
lC23:
	.ascii "3-8-1\0"
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
	mov	w0, 8
	str	w0, [x29, 24]
	mov	w0, 1
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
	cmp	w0, 8
	beq	L6
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L58
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L58
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L58
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L58
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L58
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L58
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L58
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L59
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L59
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L60
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L60
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L61
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L61
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L62
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L62
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L63
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L63
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L64
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L64
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L65
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L65
L6:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L66
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L66
L59:
	nop
	b	L58
L60:
	nop
	b	L58
L61:
	nop
	b	L58
L62:
	nop
	b	L58
L63:
	nop
	b	L58
L64:
	nop
	b	L58
L65:
	nop
	b	L58
L66:
	nop
	b	L58
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L23
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L67
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L25
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L67
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L26
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L67
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L27
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L67
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L67
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L67
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L31
	b	L67
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L68
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L68
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L69
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L69
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L70
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L70
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L71
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L71
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L72
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L72
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L73
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L73
L25:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L74
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L74
L23:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L75
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L75
L68:
	nop
	b	L67
L69:
	nop
	b	L67
L70:
	nop
	b	L67
L71:
	nop
	b	L67
L72:
	nop
	b	L67
L73:
	nop
	b	L67
L74:
	nop
	b	L67
L75:
	nop
	b	L67
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L76
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L42
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L76
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L76
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L44
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L76
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L76
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L46
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L76
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L47
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L48
	b	L76
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L77
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L77
L48:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L78
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L78
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L79
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	nop
	b	L79
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L80
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
	b	L80
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L81
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
	b	L81
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L82
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
	b	L82
L42:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L83
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	nop
	b	L83
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L84
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
	b	L84
L77:
	nop
	b	L76
L78:
	nop
	b	L76
L79:
	nop
	b	L76
L80:
	nop
	b	L76
L81:
	nop
	b	L76
L82:
	nop
	b	L76
L83:
	nop
	b	L76
L84:
	nop
	b	L76
L58:
	nop
	b	L3
L67:
	nop
	b	L3
L76:
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
