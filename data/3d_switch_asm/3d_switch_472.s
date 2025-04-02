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
	.align	3
lC10:
	.ascii "1-6-1\0"
	.align	3
lC11:
	.ascii "1-6-2\0"
	.align	3
lC12:
	.ascii "1-7-1\0"
	.align	3
lC13:
	.ascii "1-7-2\0"
	.align	3
lC14:
	.ascii "1-8-1\0"
	.align	3
lC15:
	.ascii "1-8-2\0"
	.align	3
lC16:
	.ascii "2-1-1\0"
	.align	3
lC17:
	.ascii "2-1-2\0"
	.align	3
lC18:
	.ascii "2-2-1\0"
	.align	3
lC19:
	.ascii "2-2-2\0"
	.align	3
lC20:
	.ascii "2-3-1\0"
	.align	3
lC21:
	.ascii "2-3-2\0"
	.align	3
lC22:
	.ascii "2-4-1\0"
	.align	3
lC23:
	.ascii "2-4-2\0"
	.align	3
lC24:
	.ascii "2-5-1\0"
	.align	3
lC25:
	.ascii "2-5-2\0"
	.align	3
lC26:
	.ascii "2-6-1\0"
	.align	3
lC27:
	.ascii "2-6-2\0"
	.align	3
lC28:
	.ascii "2-7-1\0"
	.align	3
lC29:
	.ascii "2-7-2\0"
	.align	3
lC30:
	.ascii "2-8-1\0"
	.align	3
lC31:
	.ascii "2-8-2\0"
	.align	3
lC32:
	.ascii "3-1-1\0"
	.align	3
lC33:
	.ascii "3-1-2\0"
	.align	3
lC34:
	.ascii "3-2-1\0"
	.align	3
lC35:
	.ascii "3-2-2\0"
	.align	3
lC36:
	.ascii "3-3-1\0"
	.align	3
lC37:
	.ascii "3-3-2\0"
	.align	3
lC38:
	.ascii "3-4-1\0"
	.align	3
lC39:
	.ascii "3-4-2\0"
	.align	3
lC40:
	.ascii "3-5-1\0"
	.align	3
lC41:
	.ascii "3-5-2\0"
	.align	3
lC42:
	.ascii "3-6-1\0"
	.align	3
lC43:
	.ascii "3-6-2\0"
	.align	3
lC44:
	.ascii "3-7-1\0"
	.align	3
lC45:
	.ascii "3-7-2\0"
	.align	3
lC46:
	.ascii "3-8-1\0"
	.align	3
lC47:
	.ascii "3-8-2\0"
	.align	3
lC48:
	.ascii "4-1-1\0"
	.align	3
lC49:
	.ascii "4-1-2\0"
	.align	3
lC50:
	.ascii "4-2-1\0"
	.align	3
lC51:
	.ascii "4-2-2\0"
	.align	3
lC52:
	.ascii "4-3-1\0"
	.align	3
lC53:
	.ascii "4-3-2\0"
	.align	3
lC54:
	.ascii "4-4-1\0"
	.align	3
lC55:
	.ascii "4-4-2\0"
	.align	3
lC56:
	.ascii "4-5-1\0"
	.align	3
lC57:
	.ascii "4-5-2\0"
	.align	3
lC58:
	.ascii "4-6-1\0"
	.align	3
lC59:
	.ascii "4-6-2\0"
	.align	3
lC60:
	.ascii "4-7-1\0"
	.align	3
lC61:
	.ascii "4-7-2\0"
	.align	3
lC62:
	.ascii "4-8-1\0"
	.align	3
lC63:
	.ascii "4-8-2\0"
	.align	3
lC64:
	.ascii "5-1-1\0"
	.align	3
lC65:
	.ascii "5-1-2\0"
	.align	3
lC66:
	.ascii "5-2-1\0"
	.align	3
lC67:
	.ascii "5-2-2\0"
	.align	3
lC68:
	.ascii "5-3-1\0"
	.align	3
lC69:
	.ascii "5-3-2\0"
	.align	3
lC70:
	.ascii "5-4-1\0"
	.align	3
lC71:
	.ascii "5-4-2\0"
	.align	3
lC72:
	.ascii "5-5-1\0"
	.align	3
lC73:
	.ascii "5-5-2\0"
	.align	3
lC74:
	.ascii "5-6-1\0"
	.align	3
lC75:
	.ascii "5-6-2\0"
	.align	3
lC76:
	.ascii "5-7-1\0"
	.align	3
lC77:
	.ascii "5-7-2\0"
	.align	3
lC78:
	.ascii "5-8-1\0"
	.align	3
lC79:
	.ascii "5-8-2\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 5
	str	w0, [x29, 28]
	mov	w0, 8
	str	w0, [x29, 24]
	mov	w0, 2
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L7
	b	L3
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L174
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L174
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L174
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L174
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L174
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L174
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L16
	b	L174
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L18
	b	L9
L17:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L19
L18:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
L19:
	b	L9
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L21
	b	L9
L20:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L22:
	b	L9
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L9
L23:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L25
L24:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
L25:
	b	L9
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L27
	b	L9
L26:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L28
L27:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L28:
	b	L9
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L30
	b	L9
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
	b	L9
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L32
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L33
	b	L9
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
	b	L9
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L35
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L36
	b	L9
L35:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L37
L36:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L37:
	b	L9
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L39
	b	L175
L38:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L40
L39:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L40:
L175:
	nop
L9:
	b	L174
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L41
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L176
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L176
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L44
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L176
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L176
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L46
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L176
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L47
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L176
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L48
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L49
	b	L176
L48:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L51
	b	L42
L50:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L52
L51:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L52:
	b	L42
L49:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L54
	b	L42
L53:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L55
L54:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L55:
	b	L42
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L57
	b	L42
L56:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L58
L57:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L58:
	b	L42
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L42
L59:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L61
L60:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L61:
	b	L42
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L63
	b	L42
L62:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L64
L63:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L64:
	b	L42
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L66
	b	L42
L65:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L67
L66:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L67:
	b	L42
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L69
	b	L42
L68:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L70
L69:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L70:
	b	L42
L41:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L177
L71:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L73
L72:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L73:
L177:
	nop
L42:
	b	L176
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L74
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L178
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L76
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L178
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L178
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L78
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L178
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L79
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L178
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L80
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L178
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L81
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L82
	b	L178
L81:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L83
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L84
	b	L75
L83:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L85
L84:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
L85:
	b	L75
L82:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L86
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L87
	b	L75
L86:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L88
L87:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L88:
	b	L75
L80:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L89
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L90
	b	L75
L89:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L91
L90:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
L91:
	b	L75
L79:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L92
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L93
	b	L75
L92:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L94
L93:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L94:
	b	L75
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L96
	b	L75
L95:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L97
L96:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L97:
	b	L75
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L98
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L99
	b	L75
L98:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L100
L99:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L100:
	b	L75
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L102
	b	L75
L101:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L103
L102:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L103:
	b	L75
L74:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L104
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L105
	b	L179
L104:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L106
L105:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L106:
L179:
	nop
L75:
	b	L178
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L107
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L109
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L110
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L111
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L112
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L113
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L114
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L115
	b	L180
L114:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L116
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L117
	b	L108
L116:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L118
L117:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L118:
	b	L108
L115:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L119
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L120
	b	L108
L119:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L121
L120:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L121:
	b	L108
L113:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L122
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L123
	b	L108
L122:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L124
L123:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L124:
	b	L108
L112:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L125
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L126
	b	L108
L125:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L127
L126:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L127:
	b	L108
L111:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L128
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L129
	b	L108
L128:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L130
L129:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L130:
	b	L108
L110:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L131
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L132
	b	L108
L131:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L133
L132:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L133:
	b	L108
L109:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L134
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L135
	b	L108
L134:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L136
L135:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	nop
L136:
	b	L108
L107:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L137
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L138
	b	L181
L137:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L139
L138:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L139:
L181:
	nop
L108:
	b	L180
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L140
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L142
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L143
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L144
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L145
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L146
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L147
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L148
	b	L182
L147:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L149
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L150
	b	L141
L149:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L151
L150:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	nop
L151:
	b	L141
L148:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L152
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L153
	b	L141
L152:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L154
L153:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	nop
L154:
	b	L141
L146:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L155
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L156
	b	L141
L155:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L157
L156:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	nop
L157:
	b	L141
L145:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L158
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L159
	b	L141
L158:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L160
L159:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
L160:
	b	L141
L144:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L161
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L162
	b	L141
L161:
	adrp	x0, lC72@PAGE
	add	x0, x0, lC72@PAGEOFF;
	bl	_puts
	b	L163
L162:
	adrp	x0, lC73@PAGE
	add	x0, x0, lC73@PAGEOFF;
	bl	_puts
	nop
L163:
	b	L141
L143:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L164
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L165
	b	L141
L164:
	adrp	x0, lC74@PAGE
	add	x0, x0, lC74@PAGEOFF;
	bl	_puts
	b	L166
L165:
	adrp	x0, lC75@PAGE
	add	x0, x0, lC75@PAGEOFF;
	bl	_puts
	nop
L166:
	b	L141
L142:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L167
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L168
	b	L141
L167:
	adrp	x0, lC76@PAGE
	add	x0, x0, lC76@PAGEOFF;
	bl	_puts
	b	L169
L168:
	adrp	x0, lC77@PAGE
	add	x0, x0, lC77@PAGEOFF;
	bl	_puts
	nop
L169:
	b	L141
L140:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L170
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L171
	b	L183
L170:
	adrp	x0, lC78@PAGE
	add	x0, x0, lC78@PAGEOFF;
	bl	_puts
	b	L172
L171:
	adrp	x0, lC79@PAGE
	add	x0, x0, lC79@PAGEOFF;
	bl	_puts
	nop
L172:
L183:
	nop
L141:
	b	L182
L174:
	nop
	b	L3
L176:
	nop
	b	L3
L178:
	nop
	b	L3
L180:
	nop
	b	L3
L182:
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
