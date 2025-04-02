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
	mov	w0, 2
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
	bne	L44
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L13
	b	L45
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
L45:
	nop
	b	L44
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L46
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L47
L16:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L18
L17:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L18:
L47:
	nop
	b	L46
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L48
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L21
	b	L49
L20:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L22:
L49:
	nop
	b	L48
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L50
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L25
	b	L51
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
L51:
	nop
	b	L50
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L52
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L29
	b	L53
L28:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L30
L29:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
L30:
L53:
	nop
	b	L52
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L54
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L33
	b	L55
L32:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L34
L33:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L34:
L55:
	nop
	b	L54
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L56
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L37
	b	L57
L36:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L38
L37:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L38:
L57:
	nop
	b	L56
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	bne	L58
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L40
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L41
	b	L59
L40:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L42
L41:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L42:
L59:
	nop
	b	L58
L44:
	nop
	b	L3
L46:
	nop
	b	L3
L48:
	nop
	b	L3
L50:
	nop
	b	L3
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
