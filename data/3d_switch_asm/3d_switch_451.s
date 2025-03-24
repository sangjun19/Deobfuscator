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
	.ascii "2-1-1\0"
	.align	3
lC7:
	.ascii "2-2-1\0"
	.align	3
lC8:
	.ascii "2-3-1\0"
	.align	3
lC9:
	.ascii "2-4-1\0"
	.align	3
lC10:
	.ascii "2-5-1\0"
	.align	3
lC11:
	.ascii "2-6-1\0"
	.align	3
lC12:
	.ascii "3-1-1\0"
	.align	3
lC13:
	.ascii "3-2-1\0"
	.align	3
lC14:
	.ascii "3-3-1\0"
	.align	3
lC15:
	.ascii "3-4-1\0"
	.align	3
lC16:
	.ascii "3-5-1\0"
	.align	3
lC17:
	.ascii "3-6-1\0"
	.align	3
lC18:
	.ascii "4-1-1\0"
	.align	3
lC19:
	.ascii "4-2-1\0"
	.align	3
lC20:
	.ascii "4-3-1\0"
	.align	3
lC21:
	.ascii "4-4-1\0"
	.align	3
lC22:
	.ascii "4-5-1\0"
	.align	3
lC23:
	.ascii "4-6-1\0"
	.align	3
lC24:
	.ascii "5-1-1\0"
	.align	3
lC25:
	.ascii "5-2-1\0"
	.align	3
lC26:
	.ascii "5-3-1\0"
	.align	3
lC27:
	.ascii "5-4-1\0"
	.align	3
lC28:
	.ascii "5-5-1\0"
	.align	3
lC29:
	.ascii "5-6-1\0"
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
	mov	w0, 6
	str	w0, [x29, 24]
	mov	w0, 1
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
	cmp	w0, 6
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L74
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L74
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L74
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L74
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L14
	b	L74
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L75
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L75
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L76
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L76
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L77
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L77
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L78
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L78
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L79
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L79
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L80
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L80
L75:
	nop
	b	L74
L76:
	nop
	b	L74
L77:
	nop
	b	L74
L78:
	nop
	b	L74
L79:
	nop
	b	L74
L80:
	nop
	b	L74
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L21
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L81
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L23
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L81
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L24
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L81
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L25
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L81
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L27
	b	L81
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L82
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L82
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L83
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L83
L25:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L84
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L84
L24:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L85
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L85
L23:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L86
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L86
L21:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L87
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L87
L82:
	nop
	b	L81
L83:
	nop
	b	L81
L84:
	nop
	b	L81
L85:
	nop
	b	L81
L86:
	nop
	b	L81
L87:
	nop
	b	L81
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L88
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L88
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L37
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L88
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L38
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L88
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L40
	b	L88
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L89
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L89
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L90
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L90
L38:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L91
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L91
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L92
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L92
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L93
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L93
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L94
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L94
L89:
	nop
	b	L88
L90:
	nop
	b	L88
L91:
	nop
	b	L88
L92:
	nop
	b	L88
L93:
	nop
	b	L88
L94:
	nop
	b	L88
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L47
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L95
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L49
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L95
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L50
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L95
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L51
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L95
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L52
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L53
	b	L95
L52:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L96
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	nop
	b	L96
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L97
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
	b	L97
L51:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L98
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
	b	L98
L50:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L99
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
	b	L99
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L100
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	nop
	b	L100
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L101
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
	b	L101
L96:
	nop
	b	L95
L97:
	nop
	b	L95
L98:
	nop
	b	L95
L99:
	nop
	b	L95
L100:
	nop
	b	L95
L101:
	nop
	b	L95
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L60
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L102
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L62
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L102
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L63
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L102
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L64
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L102
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L66
	b	L102
L65:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L103
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
	b	L103
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L104
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
	b	L104
L64:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L105
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
	b	L105
L63:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L106
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
	b	L106
L62:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L107
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	nop
	b	L107
L60:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L108
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
	b	L108
L103:
	nop
	b	L102
L104:
	nop
	b	L102
L105:
	nop
	b	L102
L106:
	nop
	b	L102
L107:
	nop
	b	L102
L108:
	nop
	b	L102
L74:
	nop
	b	L3
L81:
	nop
	b	L3
L88:
	nop
	b	L3
L95:
	nop
	b	L3
L102:
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
