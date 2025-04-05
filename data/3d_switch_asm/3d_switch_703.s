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
	mov	w0, 1
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
	bne	L52
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L15
	b	L53
L14:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L13
L15:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L13
L12:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
L13:
L53:
	nop
	b	L52
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L54
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L55
L19:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L18
L20:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L18:
L55:
	nop
	b	L54
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L56
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L57
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L25
	b	L57
L24:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L23
L25:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L23
L22:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L23:
L57:
	nop
	b	L56
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L58
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L59
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L30
	b	L59
L29:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L28
L30:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L28
L27:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L28:
L59:
	nop
	b	L58
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L60
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L61
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L35
	b	L61
L34:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L33
L35:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L33
L32:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L33:
L61:
	nop
	b	L60
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L62
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L63
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L40
	b	L63
L39:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L38
L40:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L38
L37:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L38:
L63:
	nop
	b	L62
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L64
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L65
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L45
	b	L65
L44:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L43
L45:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L43
L42:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L43:
L65:
	nop
	b	L64
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L66
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L67
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L67
L49:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L48
L50:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L48
L47:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L48:
L67:
	nop
	b	L66
L52:
	nop
	b	L3
L54:
	nop
	b	L3
L56:
	nop
	b	L3
L58:
	nop
	b	L3
L60:
	nop
	b	L3
L62:
	nop
	b	L3
L64:
	nop
	b	L3
L66:
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
