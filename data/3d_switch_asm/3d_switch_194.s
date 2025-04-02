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
	.ascii "1-2-1\0"
	.align	3
lC5:
	.ascii "1-2-2\0"
	.align	3
lC6:
	.ascii "1-2-3\0"
	.align	3
lC7:
	.ascii "1-2-4\0"
	.align	3
lC8:
	.ascii "1-3-1\0"
	.align	3
lC9:
	.ascii "1-3-2\0"
	.align	3
lC10:
	.ascii "1-3-3\0"
	.align	3
lC11:
	.ascii "1-3-4\0"
	.align	3
lC12:
	.ascii "1-4-1\0"
	.align	3
lC13:
	.ascii "1-4-2\0"
	.align	3
lC14:
	.ascii "1-4-3\0"
	.align	3
lC15:
	.ascii "1-4-4\0"
	.align	3
lC16:
	.ascii "1-5-1\0"
	.align	3
lC17:
	.ascii "1-5-2\0"
	.align	3
lC18:
	.ascii "1-5-3\0"
	.align	3
lC19:
	.ascii "1-5-4\0"
	.align	3
lC20:
	.ascii "1-6-1\0"
	.align	3
lC21:
	.ascii "1-6-2\0"
	.align	3
lC22:
	.ascii "1-6-3\0"
	.align	3
lC23:
	.ascii "1-6-4\0"
	.align	3
lC24:
	.ascii "1-7-1\0"
	.align	3
lC25:
	.ascii "1-7-2\0"
	.align	3
lC26:
	.ascii "1-7-3\0"
	.align	3
lC27:
	.ascii "1-7-4\0"
	.align	3
lC28:
	.ascii "1-8-1\0"
	.align	3
lC29:
	.ascii "1-8-2\0"
	.align	3
lC30:
	.ascii "1-8-3\0"
	.align	3
lC31:
	.ascii "1-8-4\0"
	.align	3
lC32:
	.ascii "1-9-1\0"
	.align	3
lC33:
	.ascii "1-9-2\0"
	.align	3
lC34:
	.ascii "1-9-3\0"
	.align	3
lC35:
	.ascii "1-9-4\0"
	.align	3
lC36:
	.ascii "1-10-1\0"
	.align	3
lC37:
	.ascii "1-10-2\0"
	.align	3
lC38:
	.ascii "1-10-3\0"
	.align	3
lC39:
	.ascii "1-10-4\0"
	.align	3
lC40:
	.ascii "2-1-1\0"
	.align	3
lC41:
	.ascii "2-1-2\0"
	.align	3
lC42:
	.ascii "2-1-3\0"
	.align	3
lC43:
	.ascii "2-1-4\0"
	.align	3
lC44:
	.ascii "2-2-1\0"
	.align	3
lC45:
	.ascii "2-2-2\0"
	.align	3
lC46:
	.ascii "2-2-3\0"
	.align	3
lC47:
	.ascii "2-2-4\0"
	.align	3
lC48:
	.ascii "2-3-1\0"
	.align	3
lC49:
	.ascii "2-3-2\0"
	.align	3
lC50:
	.ascii "2-3-3\0"
	.align	3
lC51:
	.ascii "2-3-4\0"
	.align	3
lC52:
	.ascii "2-4-1\0"
	.align	3
lC53:
	.ascii "2-4-2\0"
	.align	3
lC54:
	.ascii "2-4-3\0"
	.align	3
lC55:
	.ascii "2-4-4\0"
	.align	3
lC56:
	.ascii "2-5-1\0"
	.align	3
lC57:
	.ascii "2-5-2\0"
	.align	3
lC58:
	.ascii "2-5-3\0"
	.align	3
lC59:
	.ascii "2-5-4\0"
	.align	3
lC60:
	.ascii "2-6-1\0"
	.align	3
lC61:
	.ascii "2-6-2\0"
	.align	3
lC62:
	.ascii "2-6-3\0"
	.align	3
lC63:
	.ascii "2-6-4\0"
	.align	3
lC64:
	.ascii "2-7-1\0"
	.align	3
lC65:
	.ascii "2-7-2\0"
	.align	3
lC66:
	.ascii "2-7-3\0"
	.align	3
lC67:
	.ascii "2-7-4\0"
	.align	3
lC68:
	.ascii "2-8-1\0"
	.align	3
lC69:
	.ascii "2-8-2\0"
	.align	3
lC70:
	.ascii "2-8-3\0"
	.align	3
lC71:
	.ascii "2-8-4\0"
	.align	3
lC72:
	.ascii "2-9-1\0"
	.align	3
lC73:
	.ascii "2-9-2\0"
	.align	3
lC74:
	.ascii "2-9-3\0"
	.align	3
lC75:
	.ascii "2-9-4\0"
	.align	3
lC76:
	.ascii "2-10-1\0"
	.align	3
lC77:
	.ascii "2-10-2\0"
	.align	3
lC78:
	.ascii "2-10-3\0"
	.align	3
lC79:
	.ascii "2-10-4\0"
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
	mov	w0, 10
	str	w0, [x29, 24]
	mov	w0, 4
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
	cmp	w0, 10
	beq	L5
	ldr	w0, [x29, 24]
	cmp	w0, 10
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L7
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L8
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L9
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L11
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L128
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L15
	b	L128
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L16
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L18
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L129
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L19
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L20
	b	L129
L19:
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	b	L17
L20:
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	b	L17
L18:
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	b	L17
L16:
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
L17:
	b	L129
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L21
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L130
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L23
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L130
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L24
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L25
	b	L130
L24:
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	b	L22
L25:
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	b	L22
L23:
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	b	L22
L21:
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
L22:
	b	L130
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L26
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L131
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L28
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L131
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L29
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L30
	b	L131
L29:
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	b	L27
L30:
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	b	L27
L28:
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	b	L27
L26:
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
L27:
	b	L131
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L31
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L132
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L33
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L132
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L34
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L35
	b	L132
L34:
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	b	L32
L35:
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	b	L32
L33:
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	b	L32
L31:
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
L32:
	b	L132
L11:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L36
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L133
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L38
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L133
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L39
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L40
	b	L133
L39:
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	b	L37
L40:
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	b	L37
L38:
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	b	L37
L36:
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
L37:
	b	L133
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L41
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L134
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L43
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L134
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L44
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L45
	b	L134
L44:
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	b	L42
L45:
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	b	L42
L43:
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	b	L42
L41:
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
L42:
	b	L134
L9:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L46
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L135
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L48
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L135
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L49
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L50
	b	L135
L49:
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	b	L47
L50:
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	b	L47
L48:
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	b	L47
L46:
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
L47:
	b	L135
L8:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L51
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L136
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L53
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L136
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L54
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L55
	b	L136
L54:
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	b	L52
L55:
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	b	L52
L53:
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	b	L52
L51:
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
L52:
	b	L136
L7:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L56
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L137
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L58
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L137
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L59
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L60
	b	L137
L59:
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	b	L57
L60:
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	b	L57
L58:
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	b	L57
L56:
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
L57:
	b	L137
L5:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L61
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L138
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L63
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L138
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L64
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L65
	b	L138
L64:
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	b	L62
L65:
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	b	L62
L63:
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	b	L62
L61:
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
L62:
	b	L138
L129:
	nop
	b	L128
L130:
	nop
	b	L128
L131:
	nop
	b	L128
L132:
	nop
	b	L128
L133:
	nop
	b	L128
L134:
	nop
	b	L128
L135:
	nop
	b	L128
L136:
	nop
	b	L128
L137:
	nop
	b	L128
L138:
	nop
	b	L128
L3:
	ldr	w0, [x29, 24]
	cmp	w0, 10
	beq	L66
	ldr	w0, [x29, 24]
	cmp	w0, 10
	bgt	L139
	ldr	w0, [x29, 24]
	cmp	w0, 9
	beq	L68
	ldr	w0, [x29, 24]
	cmp	w0, 9
	bgt	L139
	ldr	w0, [x29, 24]
	cmp	w0, 8
	beq	L69
	ldr	w0, [x29, 24]
	cmp	w0, 8
	bgt	L139
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L139
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L71
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L139
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L72
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L139
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L73
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L139
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L74
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L139
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L76
	b	L139
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L77
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L140
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L79
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L140
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L80
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L81
	b	L140
L80:
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	b	L78
L81:
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	b	L78
L79:
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	b	L78
L77:
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
L78:
	b	L140
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L82
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L141
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L84
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L141
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L85
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L86
	b	L141
L85:
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	b	L83
L86:
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	b	L83
L84:
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	b	L83
L82:
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
L83:
	b	L141
L74:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L87
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L142
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L89
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L142
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L90
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L91
	b	L142
L90:
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	b	L88
L91:
	adrp	x0, lC49@PAGE
	add	x0, x0, lC49@PAGEOFF;
	bl	_puts
	b	L88
L89:
	adrp	x0, lC50@PAGE
	add	x0, x0, lC50@PAGEOFF;
	bl	_puts
	b	L88
L87:
	adrp	x0, lC51@PAGE
	add	x0, x0, lC51@PAGEOFF;
	bl	_puts
	nop
L88:
	b	L142
L73:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L92
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L143
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L94
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L143
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L95
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L96
	b	L143
L95:
	adrp	x0, lC52@PAGE
	add	x0, x0, lC52@PAGEOFF;
	bl	_puts
	b	L93
L96:
	adrp	x0, lC53@PAGE
	add	x0, x0, lC53@PAGEOFF;
	bl	_puts
	b	L93
L94:
	adrp	x0, lC54@PAGE
	add	x0, x0, lC54@PAGEOFF;
	bl	_puts
	b	L93
L92:
	adrp	x0, lC55@PAGE
	add	x0, x0, lC55@PAGEOFF;
	bl	_puts
	nop
L93:
	b	L143
L72:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L97
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L144
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L99
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L144
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L100
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L101
	b	L144
L100:
	adrp	x0, lC56@PAGE
	add	x0, x0, lC56@PAGEOFF;
	bl	_puts
	b	L98
L101:
	adrp	x0, lC57@PAGE
	add	x0, x0, lC57@PAGEOFF;
	bl	_puts
	b	L98
L99:
	adrp	x0, lC58@PAGE
	add	x0, x0, lC58@PAGEOFF;
	bl	_puts
	b	L98
L97:
	adrp	x0, lC59@PAGE
	add	x0, x0, lC59@PAGEOFF;
	bl	_puts
	nop
L98:
	b	L144
L71:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L102
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L145
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L104
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L145
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L105
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L106
	b	L145
L105:
	adrp	x0, lC60@PAGE
	add	x0, x0, lC60@PAGEOFF;
	bl	_puts
	b	L103
L106:
	adrp	x0, lC61@PAGE
	add	x0, x0, lC61@PAGEOFF;
	bl	_puts
	b	L103
L104:
	adrp	x0, lC62@PAGE
	add	x0, x0, lC62@PAGEOFF;
	bl	_puts
	b	L103
L102:
	adrp	x0, lC63@PAGE
	add	x0, x0, lC63@PAGEOFF;
	bl	_puts
	nop
L103:
	b	L145
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L107
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L146
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L109
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L146
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L110
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L111
	b	L146
L110:
	adrp	x0, lC64@PAGE
	add	x0, x0, lC64@PAGEOFF;
	bl	_puts
	b	L108
L111:
	adrp	x0, lC65@PAGE
	add	x0, x0, lC65@PAGEOFF;
	bl	_puts
	b	L108
L109:
	adrp	x0, lC66@PAGE
	add	x0, x0, lC66@PAGEOFF;
	bl	_puts
	b	L108
L107:
	adrp	x0, lC67@PAGE
	add	x0, x0, lC67@PAGEOFF;
	bl	_puts
	nop
L108:
	b	L146
L69:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L112
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L147
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L114
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L147
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L115
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L116
	b	L147
L115:
	adrp	x0, lC68@PAGE
	add	x0, x0, lC68@PAGEOFF;
	bl	_puts
	b	L113
L116:
	adrp	x0, lC69@PAGE
	add	x0, x0, lC69@PAGEOFF;
	bl	_puts
	b	L113
L114:
	adrp	x0, lC70@PAGE
	add	x0, x0, lC70@PAGEOFF;
	bl	_puts
	b	L113
L112:
	adrp	x0, lC71@PAGE
	add	x0, x0, lC71@PAGEOFF;
	bl	_puts
	nop
L113:
	b	L147
L68:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L117
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L148
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L119
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L148
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L120
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L121
	b	L148
L120:
	adrp	x0, lC72@PAGE
	add	x0, x0, lC72@PAGEOFF;
	bl	_puts
	b	L118
L121:
	adrp	x0, lC73@PAGE
	add	x0, x0, lC73@PAGEOFF;
	bl	_puts
	b	L118
L119:
	adrp	x0, lC74@PAGE
	add	x0, x0, lC74@PAGEOFF;
	bl	_puts
	b	L118
L117:
	adrp	x0, lC75@PAGE
	add	x0, x0, lC75@PAGEOFF;
	bl	_puts
	nop
L118:
	b	L148
L66:
	ldr	w0, [x29, 20]
	cmp	w0, 4
	beq	L122
	ldr	w0, [x29, 20]
	cmp	w0, 4
	bgt	L149
	ldr	w0, [x29, 20]
	cmp	w0, 3
	beq	L124
	ldr	w0, [x29, 20]
	cmp	w0, 3
	bgt	L149
	ldr	w0, [x29, 20]
	cmp	w0, 1
	beq	L125
	ldr	w0, [x29, 20]
	cmp	w0, 2
	beq	L126
	b	L149
L125:
	adrp	x0, lC76@PAGE
	add	x0, x0, lC76@PAGEOFF;
	bl	_puts
	b	L123
L126:
	adrp	x0, lC77@PAGE
	add	x0, x0, lC77@PAGEOFF;
	bl	_puts
	b	L123
L124:
	adrp	x0, lC78@PAGE
	add	x0, x0, lC78@PAGEOFF;
	bl	_puts
	b	L123
L122:
	adrp	x0, lC79@PAGE
	add	x0, x0, lC79@PAGEOFF;
	bl	_puts
	nop
L123:
	b	L149
L140:
	nop
	b	L139
L141:
	nop
	b	L139
L142:
	nop
	b	L139
L143:
	nop
	b	L139
L144:
	nop
	b	L139
L145:
	nop
	b	L139
L146:
	nop
	b	L139
L147:
	nop
	b	L139
L148:
	nop
	b	L139
L149:
	nop
	b	L139
L128:
	nop
	b	L4
L139:
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
