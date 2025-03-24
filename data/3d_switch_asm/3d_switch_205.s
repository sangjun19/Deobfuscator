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
	.ascii "2-1-1\0"
	.align	3
lC6:
	.ascii "2-1-2\0"
	.align	3
lC7:
	.ascii "2-1-3\0"
	.align	3
lC8:
	.ascii "2-1-4\0"
	.align	3
lC9:
	.ascii "2-1-5\0"
	.align	3
lC10:
	.ascii "3-1-1\0"
	.align	3
lC11:
	.ascii "3-1-2\0"
	.align	3
lC12:
	.ascii "3-1-3\0"
	.align	3
lC13:
	.ascii "3-1-4\0"
	.align	3
lC14:
	.ascii "3-1-5\0"
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
	mov	w0, 5
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
	bne	L28
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L7
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L29
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L9
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L29
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L10
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L29
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L11
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L12
	b	L29
L11:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L8
L12:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L8
L10:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L8
L9:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L8
L7:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
L8:
L29:
	nop
	b	L28
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L30
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L31
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L31
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L31
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L19
	b	L31
L18:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L15
L19:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L15
L17:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L15
L16:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L15
L14:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L15:
L31:
	nop
	b	L30
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L32
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L33
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L33
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L33
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L26
	b	L33
L25:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L22
L26:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	b	L22
L24:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L22
L23:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
L22:
L33:
	nop
	b	L32
L28:
	nop
	b	L3
L30:
	nop
	b	L3
L32:
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
