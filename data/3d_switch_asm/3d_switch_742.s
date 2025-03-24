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
	.ascii "2-1-1\0"
	.align	3
lC11:
	.ascii "2-1-2\0"
	.align	3
lC12:
	.ascii "2-2-1\0"
	.align	3
lC13:
	.ascii "2-2-2\0"
	.align	3
lC14:
	.ascii "2-3-1\0"
	.align	3
lC15:
	.ascii "2-3-2\0"
	.align	3
lC16:
	.ascii "2-4-1\0"
	.align	3
lC17:
	.ascii "2-4-2\0"
	.align	3
lC18:
	.ascii "2-5-1\0"
	.align	3
lC19:
	.ascii "2-5-2\0"
	.align	3
lC20:
	.ascii "3-1-1\0"
	.align	3
lC21:
	.ascii "3-1-2\0"
	.align	3
lC22:
	.ascii "3-2-1\0"
	.align	3
lC23:
	.ascii "3-2-2\0"
	.align	3
lC24:
	.ascii "3-3-1\0"
	.align	3
lC25:
	.ascii "3-3-2\0"
	.align	3
lC26:
	.ascii "3-4-1\0"
	.align	3
lC27:
	.ascii "3-4-2\0"
	.align	3
lC28:
	.ascii "3-5-1\0"
	.align	3
lC29:
	.ascii "3-5-2\0"
	.align	3
lC30:
	.ascii "4-1-1\0"
	.align	3
lC31:
	.ascii "4-1-2\0"
	.align	3
lC32:
	.ascii "4-2-1\0"
	.align	3
lC33:
	.ascii "4-2-2\0"
	.align	3
lC34:
	.ascii "4-3-1\0"
	.align	3
lC35:
	.ascii "4-3-2\0"
	.align	3
lC36:
	.ascii "4-4-1\0"
	.align	3
lC37:
	.ascii "4-4-2\0"
	.align	3
lC38:
	.ascii "4-5-1\0"
	.align	3
lC39:
	.ascii "4-5-2\0"
	.align	3
lC40:
	.ascii "5-1-1\0"
	.align	3
lC41:
	.ascii "5-1-2\0"
	.align	3
lC42:
	.ascii "5-2-1\0"
	.align	3
lC43:
	.ascii "5-2-2\0"
	.align	3
lC44:
	.ascii "5-3-1\0"
	.align	3
lC45:
	.ascii "5-3-2\0"
	.align	3
lC46:
	.ascii "5-4-1\0"
	.align	3
lC47:
	.ascii "5-4-2\0"
	.align	3
lC48:
	.ascii "5-5-1\0"
	.align	3
lC49:
	.ascii "5-5-2\0"
	.align	3
lC50:
	.ascii "6-1-1\0"
	.align	3
lC51:
	.ascii "6-1-2\0"
	.align	3
lC52:
	.ascii "6-2-1\0"
	.align	3
lC53:
	.ascii "6-2-2\0"
	.align	3
lC54:
	.ascii "6-3-1\0"
	.align	3
lC55:
	.ascii "6-3-2\0"
	.align	3
lC56:
	.ascii "6-4-1\0"
	.align	3
lC57:
	.ascii "6-4-2\0"
	.align	3
lC58:
	.ascii "6-5-1\0"
	.align	3
lC59:
	.ascii "6-5-2\0"
	.align	3
lC60:
	.ascii "7-1-1\0"
	.align	3
lC61:
	.ascii "7-1-2\0"
	.align	3
lC62:
	.ascii "7-2-1\0"
	.align	3
lC63:
	.ascii "7-2-2\0"
	.align	3
lC64:
	.ascii "7-3-1\0"
	.align	3
lC65:
	.ascii "7-3-2\0"
	.align	3
lC66:
	.ascii "7-4-1\0"
	.align	3
lC67:
	.ascii "7-4-2\0"
	.align	3
lC68:
	.ascii "7-5-1\0"
	.align	3
lC69:
	.ascii "7-5-2\0"
	.align	3
lC70:
	.ascii "8-1-1\0"
	.align	3
lC71:
	.ascii "8-1-2\0"
	.align	3
lC72:
	.ascii "8-2-1\0"
	.align	3
lC73:
	.ascii "8-2-2\0"
	.align	3
lC74:
	.ascii "8-3-1\0"
	.align	3
lC75:
	.ascii "8-3-2\0"
	.align	3
lC76:
	.ascii "8-4-1\0"
	.align	3
lC77:
	.ascii "8-4-2\0"
	.align	3
lC78:
	.ascii "8-5-1\0"
	.align	3
lC79:
	.ascii "8-5-2\0"
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
	mov	w0, 5
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
	cmp	w0, 5
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L180
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L16
	b	L180
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L17
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L18
	b	L12
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
	b	L12
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L20
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L21
	b	L12
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
	b	L12
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L24
	b	L12
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
	b	L12
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L27
	b	L12
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
	b	L12
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L30
	b	L181
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
L181:
	nop
L12:
	b	L180
L10:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L32
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L34
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L35
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L182
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L36
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L37
	b	L182
L36:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L39
	b	L33
L38:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L40
L39:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L40:
	b	L33
L37:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L42
	b	L33
L41:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L43
L42:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
L43:
	b	L33
L35:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L45
	b	L33
L44:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L46
L45:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L46:
	b	L33
L34:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L47
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L48
	b	L33
L47:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L49
L48:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
L49:
	b	L33
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L50
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L51
	b	L183
L50:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L52
L51:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L52:
L183:
	nop
L33:
	b	L182
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L53
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L55
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L56
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L184
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L57
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L58
	b	L184
L57:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L54
L59:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L61
L60:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
L61:
	b	L54
L58:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L62
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L63
	b	L54
L62:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L64
L63:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L64:
	b	L54
L56:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L65
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L66
	b	L54
L65:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L67
L66:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
L67:
	b	L54
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L68
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L69
	b	L54
L68:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L70
L69:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L70:
	b	L54
L53:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L71
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L72
	b	L185
L71:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L73
L72:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
L73:
L185:
	nop
L54:
	b	L184
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L74
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L186
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L76
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L186
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L77
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L186
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L78
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L79
	b	L186
L78:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L80
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L81
	b	L75
L80:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L82
L81:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L82:
	b	L75
L79:
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
L77:
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
L76:
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
L74:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L92
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L93
	b	L187
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
L187:
	nop
L75:
	b	L186
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L95
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L188
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L97
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L188
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L98
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L188
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L99
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L100
	b	L188
L99:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L101
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L102
	b	L96
L101:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L103
L102:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
L103:
	b	L96
L100:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L104
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L105
	b	L96
L104:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L106
L105:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L106:
	b	L96
L98:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L107
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L108
	b	L96
L107:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L109
L108:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
L109:
	b	L96
L97:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L110
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L111
	b	L96
L110:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L112
L111:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L112:
	b	L96
L95:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L113
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L114
	b	L189
L113:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L115
L114:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	nop
L115:
L189:
	nop
L96:
	b	L188
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L116
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L190
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L118
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L190
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L119
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L190
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L120
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L121
	b	L190
L120:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L122
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L123
	b	L117
L122:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L124
L123:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L124:
	b	L117
L121:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L125
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L126
	b	L117
L125:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L127
L126:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	nop
L127:
	b	L117
L119:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L128
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L129
	b	L117
L128:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L130
L129:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L130:
	b	L117
L118:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L131
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L132
	b	L117
L131:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L133
L132:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	nop
L133:
	b	L117
L116:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L134
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L135
	b	L191
L134:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L136
L135:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L136:
L191:
	nop
L117:
	b	L190
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L137
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L192
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L139
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L192
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L140
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L192
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L141
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L142
	b	L192
L141:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L143
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L144
	b	L138
L143:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L145
L144:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	nop
L145:
	b	L138
L142:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L146
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L147
	b	L138
L146:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L148
L147:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L148:
	b	L138
L140:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L149
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L150
	b	L138
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
	b	L138
L139:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L152
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L153
	b	L138
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
	b	L138
L137:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L155
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L156
	b	L193
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
L193:
	nop
L138:
	b	L192
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L158
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L194
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L160
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L194
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L161
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L194
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L162
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L163
	b	L194
L162:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L164
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L165
	b	L159
L164:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L166
L165:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
L166:
	b	L159
L163:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L167
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L168
	b	L159
L167:
	adrp	x0, lC72@PAGE
	add	x0, x0, lC72@PAGEOFF;
	bl	_puts
	b	L169
L168:
	adrp	x0, lC73@PAGE
	add	x0, x0, lC73@PAGEOFF;
	bl	_puts
	nop
L169:
	b	L159
L161:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L170
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L171
	b	L159
L170:
	adrp	x0, lC74@PAGE
	add	x0, x0, lC74@PAGEOFF;
	bl	_puts
	b	L172
L171:
	adrp	x0, lC75@PAGE
	add	x0, x0, lC75@PAGEOFF;
	bl	_puts
	nop
L172:
	b	L159
L160:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L173
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L174
	b	L159
L173:
	adrp	x0, lC76@PAGE
	add	x0, x0, lC76@PAGEOFF;
	bl	_puts
	b	L175
L174:
	adrp	x0, lC77@PAGE
	add	x0, x0, lC77@PAGEOFF;
	bl	_puts
	nop
L175:
	b	L159
L158:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L176
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L177
	b	L195
L176:
	adrp	x0, lC78@PAGE
	add	x0, x0, lC78@PAGEOFF;
	bl	_puts
	b	L178
L177:
	adrp	x0, lC79@PAGE
	add	x0, x0, lC79@PAGEOFF;
	bl	_puts
	nop
L178:
L195:
	nop
L159:
	b	L194
L180:
	nop
	b	L3
L182:
	nop
	b	L3
L184:
	nop
	b	L3
L186:
	nop
	b	L3
L188:
	nop
	b	L3
L190:
	nop
	b	L3
L192:
	nop
	b	L3
L194:
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
