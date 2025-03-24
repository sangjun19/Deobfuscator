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
	.ascii "2-1-1\0"
	.align	3
lC4:
	.ascii "2-2-1\0"
	.align	3
lC5:
	.ascii "2-3-1\0"
	.align	3
lC6:
	.ascii "3-1-1\0"
	.align	3
lC7:
	.ascii "3-2-1\0"
	.align	3
lC8:
	.ascii "3-3-1\0"
	.align	3
lC9:
	.ascii "4-1-1\0"
	.align	3
lC10:
	.ascii "4-2-1\0"
	.align	3
lC11:
	.ascii "4-3-1\0"
	.align	3
lC12:
	.ascii "5-1-1\0"
	.align	3
lC13:
	.ascii "5-2-1\0"
	.align	3
lC14:
	.ascii "5-3-1\0"
	.align	3
lC15:
	.ascii "6-1-1\0"
	.align	3
lC16:
	.ascii "6-2-1\0"
	.align	3
lC17:
	.ascii "6-3-1\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 6
	str	w0, [x29, 28]
	mov	w0, 3
	str	w0, [x29, 24]
	mov	w0, 1
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L8
	b	L3
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L52
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L12
	b	L52
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L53
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L53
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L54
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L54
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L55
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L55
L53:
	nop
	b	L52
L54:
	nop
	b	L52
L55:
	nop
	b	L52
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L16
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L56
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L19
	b	L56
L18:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L57
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L57
L19:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L58
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L58
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L59
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L59
L57:
	nop
	b	L56
L58:
	nop
	b	L56
L59:
	nop
	b	L56
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L23
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L60
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L26
	b	L60
L25:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L61
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L61
L26:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L62
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L62
L23:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L63
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L63
L61:
	nop
	b	L60
L62:
	nop
	b	L60
L63:
	nop
	b	L60
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L64
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L33
	b	L64
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L65
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L65
L33:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L66
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L66
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L67
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L67
L65:
	nop
	b	L64
L66:
	nop
	b	L64
L67:
	nop
	b	L64
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L37
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L68
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L40
	b	L68
L39:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L69
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L69
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L70
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L70
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L71
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L71
L69:
	nop
	b	L68
L70:
	nop
	b	L68
L71:
	nop
	b	L68
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L44
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L72
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L47
	b	L72
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L73
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L73
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L74
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L74
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L75
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L75
L73:
	nop
	b	L72
L74:
	nop
	b	L72
L75:
	nop
	b	L72
L52:
	nop
	b	L3
L56:
	nop
	b	L3
L60:
	nop
	b	L3
L64:
	nop
	b	L3
L68:
	nop
	b	L3
L72:
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
