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
	.ascii "1-6-1\0"
	.align	3
lC6:
	.ascii "1-7-1\0"
	.align	3
lC7:
	.ascii "2-1-1\0"
	.align	3
lC8:
	.ascii "2-2-1\0"
	.align	3
lC9:
	.ascii "2-3-1\0"
	.align	3
lC10:
	.ascii "2-4-1\0"
	.align	3
lC11:
	.ascii "2-5-1\0"
	.align	3
lC12:
	.ascii "2-6-1\0"
	.align	3
lC13:
	.ascii "2-7-1\0"
	.align	3
lC14:
	.ascii "3-1-1\0"
	.align	3
lC15:
	.ascii "3-2-1\0"
	.align	3
lC16:
	.ascii "3-3-1\0"
	.align	3
lC17:
	.ascii "3-4-1\0"
	.align	3
lC18:
	.ascii "3-5-1\0"
	.align	3
lC19:
	.ascii "3-6-1\0"
	.align	3
lC20:
	.ascii "3-7-1\0"
	.align	3
lC21:
	.ascii "4-1-1\0"
	.align	3
lC22:
	.ascii "4-2-1\0"
	.align	3
lC23:
	.ascii "4-3-1\0"
	.align	3
lC24:
	.ascii "4-4-1\0"
	.align	3
lC25:
	.ascii "4-5-1\0"
	.align	3
lC26:
	.ascii "4-6-1\0"
	.align	3
lC27:
	.ascii "4-7-1\0"
	.align	3
lC28:
	.ascii "5-1-1\0"
	.align	3
lC29:
	.ascii "5-2-1\0"
	.align	3
lC30:
	.ascii "5-3-1\0"
	.align	3
lC31:
	.ascii "5-4-1\0"
	.align	3
lC32:
	.ascii "5-5-1\0"
	.align	3
lC33:
	.ascii "5-6-1\0"
	.align	3
lC34:
	.ascii "5-7-1\0"
	.align	3
lC35:
	.ascii "6-1-1\0"
	.align	3
lC36:
	.ascii "6-2-1\0"
	.align	3
lC37:
	.ascii "6-3-1\0"
	.align	3
lC38:
	.ascii "6-4-1\0"
	.align	3
lC39:
	.ascii "6-5-1\0"
	.align	3
lC40:
	.ascii "6-6-1\0"
	.align	3
lC41:
	.ascii "6-7-1\0"
	.align	3
lC42:
	.ascii "7-1-1\0"
	.align	3
lC43:
	.ascii "7-2-1\0"
	.align	3
lC44:
	.ascii "7-3-1\0"
	.align	3
lC45:
	.ascii "7-4-1\0"
	.align	3
lC46:
	.ascii "7-5-1\0"
	.align	3
lC47:
	.ascii "7-6-1\0"
	.align	3
lC48:
	.ascii "7-7-1\0"
	.text
	.align	2
	.globl _main
_main:
LFB1:
	stp	x29, x30, [sp, -32]!
LCFI0:
	mov	x29, sp
LCFI1:
	mov	w0, 7
	str	w0, [x29, 28]
	mov	w0, 7
	str	w0, [x29, 24]
	mov	w0, 1
	str	w0, [x29, 20]
	ldr	w0, [x29, 28]
	cmp	w0, 7
	beq	L2
	ldr	w0, [x29, 28]
	cmp	w0, 7
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 6
	beq	L4
	ldr	w0, [x29, 28]
	cmp	w0, 6
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 5
	beq	L5
	ldr	w0, [x29, 28]
	cmp	w0, 5
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 4
	beq	L6
	ldr	w0, [x29, 28]
	cmp	w0, 4
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 3
	beq	L7
	ldr	w0, [x29, 28]
	cmp	w0, 3
	bgt	L3
	ldr	w0, [x29, 28]
	cmp	w0, 1
	beq	L8
	ldr	w0, [x29, 28]
	cmp	w0, 2
	beq	L9
	b	L3
L8:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L10
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L12
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L13
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L14
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L15
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L116
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L16
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L17
	b	L116
L16:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L117
	adrp	x0, lC0@PAGE
	add	x0, x0, lC0@PAGEOFF;
	bl	_puts
	nop
	b	L117
L17:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L118
	adrp	x0, lC1@PAGE
	add	x0, x0, lC1@PAGEOFF;
	bl	_puts
	nop
	b	L118
L15:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L119
	adrp	x0, lC2@PAGE
	add	x0, x0, lC2@PAGEOFF;
	bl	_puts
	nop
	b	L119
L14:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L120
	adrp	x0, lC3@PAGE
	add	x0, x0, lC3@PAGEOFF;
	bl	_puts
	nop
	b	L120
L13:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L121
	adrp	x0, lC4@PAGE
	add	x0, x0, lC4@PAGEOFF;
	bl	_puts
	nop
	b	L121
L12:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L122
	adrp	x0, lC5@PAGE
	add	x0, x0, lC5@PAGEOFF;
	bl	_puts
	nop
	b	L122
L10:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L123
	adrp	x0, lC6@PAGE
	add	x0, x0, lC6@PAGEOFF;
	bl	_puts
	nop
	b	L123
L117:
	nop
	b	L116
L118:
	nop
	b	L116
L119:
	nop
	b	L116
L120:
	nop
	b	L116
L121:
	nop
	b	L116
L122:
	nop
	b	L116
L123:
	nop
	b	L116
L9:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L25
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L27
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L28
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L29
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L30
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L124
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L31
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L32
	b	L124
L31:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L125
	adrp	x0, lC7@PAGE
	add	x0, x0, lC7@PAGEOFF;
	bl	_puts
	nop
	b	L125
L32:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L126
	adrp	x0, lC8@PAGE
	add	x0, x0, lC8@PAGEOFF;
	bl	_puts
	nop
	b	L126
L30:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L127
	adrp	x0, lC9@PAGE
	add	x0, x0, lC9@PAGEOFF;
	bl	_puts
	nop
	b	L127
L29:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L128
	adrp	x0, lC10@PAGE
	add	x0, x0, lC10@PAGEOFF;
	bl	_puts
	nop
	b	L128
L28:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L129
	adrp	x0, lC11@PAGE
	add	x0, x0, lC11@PAGEOFF;
	bl	_puts
	nop
	b	L129
L27:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L130
	adrp	x0, lC12@PAGE
	add	x0, x0, lC12@PAGEOFF;
	bl	_puts
	nop
	b	L130
L25:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L131
	adrp	x0, lC13@PAGE
	add	x0, x0, lC13@PAGEOFF;
	bl	_puts
	nop
	b	L131
L125:
	nop
	b	L124
L126:
	nop
	b	L124
L127:
	nop
	b	L124
L128:
	nop
	b	L124
L129:
	nop
	b	L124
L130:
	nop
	b	L124
L131:
	nop
	b	L124
L7:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L40
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L132
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L42
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L132
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L43
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L132
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L44
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L132
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L45
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L132
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L46
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L47
	b	L132
L46:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L133
	adrp	x0, lC14@PAGE
	add	x0, x0, lC14@PAGEOFF;
	bl	_puts
	nop
	b	L133
L47:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L134
	adrp	x0, lC15@PAGE
	add	x0, x0, lC15@PAGEOFF;
	bl	_puts
	nop
	b	L134
L45:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L135
	adrp	x0, lC16@PAGE
	add	x0, x0, lC16@PAGEOFF;
	bl	_puts
	nop
	b	L135
L44:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L136
	adrp	x0, lC17@PAGE
	add	x0, x0, lC17@PAGEOFF;
	bl	_puts
	nop
	b	L136
L43:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L137
	adrp	x0, lC18@PAGE
	add	x0, x0, lC18@PAGEOFF;
	bl	_puts
	nop
	b	L137
L42:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L138
	adrp	x0, lC19@PAGE
	add	x0, x0, lC19@PAGEOFF;
	bl	_puts
	nop
	b	L138
L40:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L139
	adrp	x0, lC20@PAGE
	add	x0, x0, lC20@PAGEOFF;
	bl	_puts
	nop
	b	L139
L133:
	nop
	b	L132
L134:
	nop
	b	L132
L135:
	nop
	b	L132
L136:
	nop
	b	L132
L137:
	nop
	b	L132
L138:
	nop
	b	L132
L139:
	nop
	b	L132
L6:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L55
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L57
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L58
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L59
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L60
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L140
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L61
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L62
	b	L140
L61:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L141
	adrp	x0, lC21@PAGE
	add	x0, x0, lC21@PAGEOFF;
	bl	_puts
	nop
	b	L141
L62:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L142
	adrp	x0, lC22@PAGE
	add	x0, x0, lC22@PAGEOFF;
	bl	_puts
	nop
	b	L142
L60:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L143
	adrp	x0, lC23@PAGE
	add	x0, x0, lC23@PAGEOFF;
	bl	_puts
	nop
	b	L143
L59:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L144
	adrp	x0, lC24@PAGE
	add	x0, x0, lC24@PAGEOFF;
	bl	_puts
	nop
	b	L144
L58:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L145
	adrp	x0, lC25@PAGE
	add	x0, x0, lC25@PAGEOFF;
	bl	_puts
	nop
	b	L145
L57:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L146
	adrp	x0, lC26@PAGE
	add	x0, x0, lC26@PAGEOFF;
	bl	_puts
	nop
	b	L146
L55:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L147
	adrp	x0, lC27@PAGE
	add	x0, x0, lC27@PAGEOFF;
	bl	_puts
	nop
	b	L147
L141:
	nop
	b	L140
L142:
	nop
	b	L140
L143:
	nop
	b	L140
L144:
	nop
	b	L140
L145:
	nop
	b	L140
L146:
	nop
	b	L140
L147:
	nop
	b	L140
L5:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L70
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L72
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L73
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L74
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L75
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L148
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L76
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L77
	b	L148
L76:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L149
	adrp	x0, lC28@PAGE
	add	x0, x0, lC28@PAGEOFF;
	bl	_puts
	nop
	b	L149
L77:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L150
	adrp	x0, lC29@PAGE
	add	x0, x0, lC29@PAGEOFF;
	bl	_puts
	nop
	b	L150
L75:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L151
	adrp	x0, lC30@PAGE
	add	x0, x0, lC30@PAGEOFF;
	bl	_puts
	nop
	b	L151
L74:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L152
	adrp	x0, lC31@PAGE
	add	x0, x0, lC31@PAGEOFF;
	bl	_puts
	nop
	b	L152
L73:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L153
	adrp	x0, lC32@PAGE
	add	x0, x0, lC32@PAGEOFF;
	bl	_puts
	nop
	b	L153
L72:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L154
	adrp	x0, lC33@PAGE
	add	x0, x0, lC33@PAGEOFF;
	bl	_puts
	nop
	b	L154
L70:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L155
	adrp	x0, lC34@PAGE
	add	x0, x0, lC34@PAGEOFF;
	bl	_puts
	nop
	b	L155
L149:
	nop
	b	L148
L150:
	nop
	b	L148
L151:
	nop
	b	L148
L152:
	nop
	b	L148
L153:
	nop
	b	L148
L154:
	nop
	b	L148
L155:
	nop
	b	L148
L4:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L85
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L87
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L88
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L89
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L90
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L156
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L91
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L92
	b	L156
L91:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L157
	adrp	x0, lC35@PAGE
	add	x0, x0, lC35@PAGEOFF;
	bl	_puts
	nop
	b	L157
L92:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L158
	adrp	x0, lC36@PAGE
	add	x0, x0, lC36@PAGEOFF;
	bl	_puts
	nop
	b	L158
L90:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L159
	adrp	x0, lC37@PAGE
	add	x0, x0, lC37@PAGEOFF;
	bl	_puts
	nop
	b	L159
L89:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L160
	adrp	x0, lC38@PAGE
	add	x0, x0, lC38@PAGEOFF;
	bl	_puts
	nop
	b	L160
L88:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L161
	adrp	x0, lC39@PAGE
	add	x0, x0, lC39@PAGEOFF;
	bl	_puts
	nop
	b	L161
L87:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L162
	adrp	x0, lC40@PAGE
	add	x0, x0, lC40@PAGEOFF;
	bl	_puts
	nop
	b	L162
L85:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L163
	adrp	x0, lC41@PAGE
	add	x0, x0, lC41@PAGEOFF;
	bl	_puts
	nop
	b	L163
L157:
	nop
	b	L156
L158:
	nop
	b	L156
L159:
	nop
	b	L156
L160:
	nop
	b	L156
L161:
	nop
	b	L156
L162:
	nop
	b	L156
L163:
	nop
	b	L156
L2:
	ldr	w0, [x29, 24]
	cmp	w0, 7
	beq	L100
	ldr	w0, [x29, 24]
	cmp	w0, 7
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 6
	beq	L102
	ldr	w0, [x29, 24]
	cmp	w0, 6
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 5
	beq	L103
	ldr	w0, [x29, 24]
	cmp	w0, 5
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 4
	beq	L104
	ldr	w0, [x29, 24]
	cmp	w0, 4
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 3
	beq	L105
	ldr	w0, [x29, 24]
	cmp	w0, 3
	bgt	L164
	ldr	w0, [x29, 24]
	cmp	w0, 1
	beq	L106
	ldr	w0, [x29, 24]
	cmp	w0, 2
	beq	L107
	b	L164
L106:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L165
	adrp	x0, lC42@PAGE
	add	x0, x0, lC42@PAGEOFF;
	bl	_puts
	nop
	b	L165
L107:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L166
	adrp	x0, lC43@PAGE
	add	x0, x0, lC43@PAGEOFF;
	bl	_puts
	nop
	b	L166
L105:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L167
	adrp	x0, lC44@PAGE
	add	x0, x0, lC44@PAGEOFF;
	bl	_puts
	nop
	b	L167
L104:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L168
	adrp	x0, lC45@PAGE
	add	x0, x0, lC45@PAGEOFF;
	bl	_puts
	nop
	b	L168
L103:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L169
	adrp	x0, lC46@PAGE
	add	x0, x0, lC46@PAGEOFF;
	bl	_puts
	nop
	b	L169
L102:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L170
	adrp	x0, lC47@PAGE
	add	x0, x0, lC47@PAGEOFF;
	bl	_puts
	nop
	b	L170
L100:
	ldr	w0, [x29, 20]
	cmp	w0, 1
	bne	L171
	adrp	x0, lC48@PAGE
	add	x0, x0, lC48@PAGEOFF;
	bl	_puts
	nop
	b	L171
L165:
	nop
	b	L164
L166:
	nop
	b	L164
L167:
	nop
	b	L164
L168:
	nop
	b	L164
L169:
	nop
	b	L164
L170:
	nop
	b	L164
L171:
	nop
	b	L164
L116:
	nop
	b	L3
L124:
	nop
	b	L3
L132:
	nop
	b	L3
L140:
	nop
	b	L3
L148:
	nop
	b	L3
L156:
	nop
	b	L3
L164:
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
