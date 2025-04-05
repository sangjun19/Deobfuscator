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
	.ascii "1-1-8\0"
	.align	3
lC8:
	.ascii "1-1-9\0"
	.align	3
lC9:
	.ascii "1-2-1\0"
	.align	3
lC10:
	.ascii "1-2-2\0"
	.align	3
lC11:
	.ascii "1-2-3\0"
	.align	3
lC12:
	.ascii "1-2-4\0"
	.align	3
lC13:
	.ascii "1-2-5\0"
	.align	3
lC14:
	.ascii "1-2-6\0"
	.align	3
lC15:
	.ascii "1-2-7\0"
	.align	3
lC16:
	.ascii "1-2-8\0"
	.align	3
lC17:
	.ascii "1-2-9\0"
	.align	3
lC18:
	.ascii "2-1-1\0"
	.align	3
lC19:
	.ascii "2-1-2\0"
	.align	3
lC20:
	.ascii "2-1-3\0"
	.align	3
lC21:
	.ascii "2-1-4\0"
	.align	3
lC22:
	.ascii "2-1-5\0"
	.align	3
lC23:
	.ascii "2-1-6\0"
	.align	3
lC24:
	.ascii "2-1-7\0"
	.align	3
lC25:
	.ascii "2-1-8\0"
	.align	3
lC26:
	.ascii "2-1-9\0"
	.align	3
lC27:
	.ascii "2-2-1\0"
	.align	3
lC28:
	.ascii "2-2-2\0"
	.align	3
lC29:
	.ascii "2-2-3\0"
	.align	3
lC30:
	.ascii "2-2-4\0"
	.align	3
lC31:
	.ascii "2-2-5\0"
	.align	3
lC32:
	.ascii "2-2-6\0"
	.align	3
lC33:
	.ascii "2-2-7\0"
	.align	3
lC34:
	.ascii "2-2-8\0"
	.align	3
lC35:
	.ascii "2-2-9\0"
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
	mov	w0, 2
	str	w0, [x29, 24]
	mov	w0, 9
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
	cmp	w0, 1
	beq	L5
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L6
	b	L4
L5:
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L8
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L52
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L10
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L52
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L11
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L52
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L12
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L52
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L13
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L52
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L14
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L52
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L15
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L52
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L17
	b	L52
L16:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L9
L17:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L9
L15:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L9
L14:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	b	L9
L13:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L9
L12:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L9
L11:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L9
L10:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	b	L9
L8:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
L9:
	b	L52
L6:
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L22
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L25
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L53
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L27
	b	L53
L26:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L19
L27:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L19
L25:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	b	L19
L24:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L19
L23:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L19
L22:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L19
L21:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	b	L19
L20:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L19
L18:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L19:
	b	L53
L52:
	nop
	b	L4
L53:
	nop
	b	L4
L3:
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L29
	b	L56
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L54
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L54
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L54
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L54
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L54
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L37
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L54
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L54
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L40
	b	L54
L39:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L32
L40:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	b	L32
L38:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L32
L37:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L32
L36:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L32
L35:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	b	L32
L34:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L32
L33:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L32
L31:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
L32:
	b	L54
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 9
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 9
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 8
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 8
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 7
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 7
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 6
	beq	L45
	ldr	w0, [x29, 20]
	cmp	w0, 6
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 5
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 5
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L55
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L55
L49:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	b	L42
L50:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L42
L48:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	b	L42
L47:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L42
L46:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	b	L42
L45:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L42
L44:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L42
L43:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L42
L41:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L42:
	b	L55
L54:
	nop
	b	L56
L55:
	nop
L56:
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
