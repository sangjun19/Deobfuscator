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
	.ascii "1-3-1\0"
	.align	3
lC5:
	.ascii "1-3-2\0"
	.align	3
lC6:
	.ascii "1-4-1\0"
	.align	3
lC7:
	.ascii "1-4-2\0"
	.align	3
lC8:
	.ascii "1-5-1\0"
	.align	3
lC9:
	.ascii "1-5-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 1
	str	w0, [x29, 28]
	mov	w0, 5
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 1
	bne	L2
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L3
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L25
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L5
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L25
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L6
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L25
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L8
	b	L25
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L9
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L10
	b	L4
L9:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L11
L10:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L11:
	b	L4
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L13
	b	L4
L12:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L14
L13:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L14:
	b	L4
L6:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L16
	b	L4
L15:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L17
L16:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L17:
	b	L4
L5:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L19
	b	L4
L18:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L20
L19:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L20:
	b	L4
L3:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L22
	b	L26
L21:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L23
L22:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L23:
L26:
	nop
L4:
L25:
	nop
L2:
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
