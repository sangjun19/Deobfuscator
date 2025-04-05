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
	.ascii "2-1-1\0"
	.align	3
lC3:
	.ascii "2-1-2\0"
	.align	3
lC4:
	.ascii "3-1-1\0"
	.align	3
lC5:
	.ascii "3-1-2\0"
	.align	3
lC6:
	.ascii "4-1-1\0"
	.align	3
lC7:
	.ascii "4-1-2\0"
	.align	3
lC8:
	.ascii "5-1-1\0"
	.align	3
lC9:
	.ascii "5-1-2\0"
	.align	3
lC10:
	.ascii "6-1-1\0"
	.align	3
lC11:
	.ascii "6-1-2\0"
	.align	3
lC12:
	.ascii "7-1-1\0"
	.align	3
lC13:
	.ascii "7-1-2\0"
	.align	3
lC14:
	.ascii "8-1-1\0"
	.align	3
lC15:
	.ascii "8-1-2\0"
	.align	3
lC16:
	.ascii "9-1-1\0"
	.align	3
lC17:
	.ascii "9-1-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 9
	str	w0, [x29, 28]
	mov	w0, 1
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 9
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 9
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 8
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 8
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L9
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L10
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L11
	b	L3
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L49
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L14
	b	L50
L13:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L15
L14:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L15:
L50:
	nop
	b	L49
L11:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L51
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L18
	b	L52
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
L52:
	nop
	b	L51
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L53
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L54
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
L54:
	nop
	b	L53
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L55
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L56
L25:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L27
L26:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L27:
L56:
	nop
	b	L55
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L57
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L30
	b	L58
L29:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L31
L30:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L31:
L58:
	nop
	b	L57
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L59
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L34
	b	L60
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
L60:
	nop
	b	L59
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L61
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L38
	b	L62
L37:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L39
L38:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L39:
L62:
	nop
	b	L61
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L63
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L42
	b	L64
L41:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L43
L42:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L43:
L64:
	nop
	b	L63
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L65
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L46
	b	L66
L45:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L47
L46:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L47:
L66:
	nop
	b	L65
L49:
	nop
	b	L3
L51:
	nop
	b	L3
L53:
	nop
	b	L3
L55:
	nop
	b	L3
L57:
	nop
	b	L3
L59:
	nop
	b	L3
L61:
	nop
	b	L3
L63:
	nop
	b	L3
L65:
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
