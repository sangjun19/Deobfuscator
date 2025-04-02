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
	mov	w0, 2
	str	w0, [x29, 24]
	mov	w0, 2
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
	cmp	w0, 1
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L10
	b	L3
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L13
	b	L11
L12:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L14
L13:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L14:
	b	L11
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L64
L15:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L17
L16:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L17:
L64:
	nop
L11:
	b	L3
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L19
	b	L3
L18:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L20
L21:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L23
L22:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L23:
	b	L20
L19:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L25
	b	L65
L24:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L26
L25:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L26:
L65:
	nop
L20:
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L28
	b	L3
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L31
	b	L29
L30:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L32
L31:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L32:
	b	L29
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L34
	b	L66
L33:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L35
L34:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L35:
L66:
	nop
L29:
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L37
	b	L3
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L40
	b	L38
L39:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L41
L40:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L41:
	b	L38
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L42
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L43
	b	L67
L42:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L44
L43:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L44:
L67:
	nop
L38:
	b	L3
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L46
	b	L3
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L49
	b	L47
L48:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L50
L49:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L50:
	b	L47
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L52
	b	L68
L51:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L53
L52:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L53:
L68:
	nop
L47:
	b	L3
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L55
	b	L70
L54:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L58
	b	L56
L57:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L59
L58:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L59:
	b	L56
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L60
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L61
	b	L69
L60:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L62
L61:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L62:
L69:
	nop
L56:
L70:
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
