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
	.ascii "2-1-1\0"
	.align	3
lC4:
	.ascii "2-1-2\0"
	.align	3
lC5:
	.ascii "2-1-3\0"
	.align	3
lC6:
	.ascii "3-1-1\0"
	.align	3
lC7:
	.ascii "3-1-2\0"
	.align	3
lC8:
	.ascii "3-1-3\0"
	.align	3
lC9:
	.ascii "4-1-1\0"
	.align	3
lC10:
	.ascii "4-1-2\0"
	.align	3
lC11:
	.ascii "4-1-3\0"
	.align	3
lC12:
	.ascii "5-1-1\0"
	.align	3
lC13:
	.ascii "5-1-2\0"
	.align	3
lC14:
	.ascii "5-1-3\0"
	.align	3
lC15:
	.ascii "6-1-1\0"
	.align	3
lC16:
	.ascii "6-1-2\0"
	.align	3
lC17:
	.ascii "6-1-3\0"
	.align	3
lC18:
	.ascii "7-1-1\0"
	.align	3
lC19:
	.ascii "7-1-2\0"
	.align	3
lC20:
	.ascii "7-1-3\0"
	.align	3
lC21:
	.ascii "8-1-1\0"
	.align	3
lC22:
	.ascii "8-1-2\0"
	.align	3
lC23:
	.ascii "8-1-3\0"
	.align	3
lC24:
	.ascii "9-1-1\0"
	.align	3
lC25:
	.ascii "9-1-2\0"
	.align	3
lC26:
	.ascii "9-1-3\0"
	.align	3
lC27:
	.ascii "10-1-1\0"
	.align	3
lC28:
	.ascii "10-1-2\0"
	.align	3
lC29:
	.ascii "10-1-3\0"
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
	mov	w0, 1
	str	w0, [x29, 24]
	mov	w0, 3
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
	bne	L64
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L65
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L65
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
L65:
	nop
	b	L64
L12:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L66
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L67
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L67
L21:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L20
L22:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L20
L19:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L20:
L67:
	nop
	b	L66
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L68
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L69
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L27
	b	L69
L26:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L25
L27:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L25
L24:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L25:
L69:
	nop
	b	L68
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L70
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L71
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L71
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
L71:
	nop
	b	L70
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L72
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L73
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L37
	b	L73
L36:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L35
L37:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L35
L34:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L35:
L73:
	nop
	b	L72
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L74
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L75
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L42
	b	L75
L41:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L40
L42:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L40
L39:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L40:
L75:
	nop
	b	L74
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L76
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L77
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L47
	b	L77
L46:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L45
L47:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L45
L44:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L45:
L77:
	nop
	b	L76
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L78
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L79
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L52
	b	L79
L51:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L50
L52:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L50
L49:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L50:
L79:
	nop
	b	L78
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L80
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L54
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L81
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L57
	b	L81
L56:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L55
L57:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L55
L54:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
L55:
L81:
	nop
	b	L80
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L82
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L83
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L62
	b	L83
L61:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L60
L62:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L60
L59:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L60:
L83:
	nop
	b	L82
L64:
	nop
	b	L3
L66:
	nop
	b	L3
L68:
	nop
	b	L3
L70:
	nop
	b	L3
L72:
	nop
	b	L3
L74:
	nop
	b	L3
L76:
	nop
	b	L3
L78:
	nop
	b	L3
L80:
	nop
	b	L3
L82:
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
