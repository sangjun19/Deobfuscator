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
	.ascii "1-1-5\0"
	.align	3
lC5:
	.ascii "1-1-6\0"
	.align	3
lC6:
	.ascii "1-1-7\0"
	.align	3
lC7:
	.ascii "2-1-1\0"
	.align	3
lC8:
	.ascii "2-1-2\0"
	.align	3
lC9:
	.ascii "2-1-3\0"
	.align	3
lC10:
	.ascii "2-1-4\0"
	.align	3
lC11:
	.ascii "2-1-5\0"
	.align	3
lC12:
	.ascii "2-1-6\0"
	.align	3
lC13:
	.ascii "2-1-7\0"
	.align	3
lC14:
	.ascii "3-1-1\0"
	.align	3
lC15:
	.ascii "3-1-2\0"
	.align	3
lC16:
	.ascii "3-1-3\0"
	.align	3
lC17:
	.ascii "3-1-4\0"
	.align	3
lC18:
	.ascii "3-1-5\0"
	.align	3
lC19:
	.ascii "3-1-6\0"
	.align	3
lC20:
	.ascii "3-1-7\0"
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
	mov	w0, 1
	str	w0, [x29, 24]
	mov	w0, 7
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
	cmp	w0, 1
	bne	L34
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L7
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L35
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L9
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L35
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L10
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L35
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L11
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L35
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L35
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L14
	b	L35
L13:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L8
L14:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L8
L12:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L8
L11:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L8
L10:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L8
L9:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L8
L7:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
L8:
L35:
	nop
	b	L34
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L36
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L37
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L37
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L37
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L37
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L37
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L23
	b	L37
L22:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L17
L23:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L17
L21:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L17
L20:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L17
L19:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	b	L17
L18:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L17
L16:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L17:
L37:
	nop
	b	L36
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L38
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L39
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L27
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L39
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L39
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L39
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L39
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L32
	b	L39
L31:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L26
L32:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L26
L30:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L26
L29:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L26
L28:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L26
L27:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L26
L25:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
L26:
L39:
	nop
	b	L38
L34:
	nop
	b	L3
L36:
	nop
	b	L3
L38:
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
