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
	.ascii "2-1-1\0"
	.align	3
lC6:
	.ascii "2-2-1\0"
	.align	3
lC7:
	.ascii "2-3-1\0"
	.align	3
lC8:
	.ascii "2-4-1\0"
	.align	3
lC9:
	.ascii "2-5-1\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 2
	str	w0, [x29, 28]
	mov	w0, 5
	str	w0, [x29, 24]
	mov	w0, 1
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L3
	b	L4
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L5
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L28
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L28
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L28
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L10
	b	L28
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L29
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L29
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L30
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L30
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L31
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L31
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L32
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L32
L5:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L33
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L33
L29:
	nop
	b	L28
L30:
	nop
	b	L28
L31:
	nop
	b	L28
L32:
	nop
	b	L28
L33:
	nop
	b	L28
L3:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L16
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L34
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L18
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L34
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L19
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L34
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L20
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L21
	b	L34
L20:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L35
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L35
L21:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L36
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L36
L19:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L37
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L37
L18:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L38
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L38
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L39
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L39
L35:
	nop
	b	L34
L36:
	nop
	b	L34
L37:
	nop
	b	L34
L38:
	nop
	b	L34
L39:
	nop
	b	L34
L28:
	nop
	b	L4
L34:
	nop
L4:
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
