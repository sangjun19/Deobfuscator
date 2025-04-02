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
	.ascii "1-2-1\0"
	.align	3
lC4:
	.ascii "1-2-2\0"
	.align	3
lC5:
	.ascii "1-2-3\0"
	.align	3
lC6:
	.ascii "2-1-1\0"
	.align	3
lC7:
	.ascii "2-1-2\0"
	.align	3
lC8:
	.ascii "2-1-3\0"
	.align	3
lC9:
	.ascii "2-2-1\0"
	.align	3
lC10:
	.ascii "2-2-2\0"
	.align	3
lC11:
	.ascii "2-2-3\0"
	.align	3
lC12:
	.ascii "3-1-1\0"
	.align	3
lC13:
	.ascii "3-1-2\0"
	.align	3
lC14:
	.ascii "3-1-3\0"
	.align	3
lC15:
	.ascii "3-2-1\0"
	.align	3
lC16:
	.ascii "3-2-2\0"
	.align	3
lC17:
	.ascii "3-2-3\0"
	.align	3
lC18:
	.ascii "4-1-1\0"
	.align	3
lC19:
	.ascii "4-1-2\0"
	.align	3
lC20:
	.ascii "4-1-3\0"
	.align	3
lC21:
	.ascii "4-2-1\0"
	.align	3
lC22:
	.ascii "4-2-2\0"
	.align	3
lC23:
	.ascii "4-2-3\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 4
	str	w0, [x29, 28]
	mov	w0, 2
	str	w0, [x29, 24]
	mov	w0, 3
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L6
	b	L3
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L8
	b	L3
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L10
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L52
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L13
	b	L52
L12:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L11
L13:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L11
L10:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
L11:
	b	L52
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L53
L16:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L15
L17:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L15
L14:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L15:
	b	L53
L52:
	nop
	b	L3
L53:
	nop
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L19
	b	L3
L18:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L54
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L54
L23:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L22
L24:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L22:
	b	L54
L19:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L28
	b	L55
L27:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L26
L28:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L26
L25:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L26:
	b	L55
L54:
	nop
	b	L3
L55:
	nop
	b	L3
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L30
	b	L3
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L56
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L35
	b	L56
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
	b	L56
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L57
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L39
	b	L57
L38:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L37
L39:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L37
L36:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L37:
	b	L57
L56:
	nop
	b	L3
L57:
	nop
	b	L3
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L41
	b	L60
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L58
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L46
	b	L58
L45:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L44
L46:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L44
L43:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L44:
	b	L58
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L59
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L59
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
	b	L59
L58:
	nop
	b	L60
L59:
	nop
L60:
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
