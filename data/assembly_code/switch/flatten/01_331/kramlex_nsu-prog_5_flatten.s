	.file	"kramlex_nsu-prog_5_flatten.c"
	.text
	.globl	_TIG_IZ_kZqg_argv
	.bss
	.align 8
	.type	_TIG_IZ_kZqg_argv, @object
	.size	_TIG_IZ_kZqg_argv, 8
_TIG_IZ_kZqg_argv:
	.zero	8
	.globl	_TIG_IZ_kZqg_envp
	.align 8
	.type	_TIG_IZ_kZqg_envp, @object
	.size	_TIG_IZ_kZqg_envp, 8
_TIG_IZ_kZqg_envp:
	.zero	8
	.globl	_TIG_IZ_kZqg_argc
	.align 4
	.type	_TIG_IZ_kZqg_argc, @object
	.size	_TIG_IZ_kZqg_argc, 4
_TIG_IZ_kZqg_argc:
	.zero	4
	.text
	.globl	char_to_int
	.type	char_to_int, @function
char_to_int:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$15, -8(%rbp)
.L44:
	cmpq	$18, -8(%rbp)
	ja	.L45
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	movl	$12, %eax
	jmp	.L23
.L18:
	movl	$6, %eax
	jmp	.L23
.L8:
	movl	$1, %eax
	jmp	.L23
.L7:
	movsbl	-20(%rbp), %eax
	subl	$48, %eax
	cmpl	$22, %eax
	ja	.L24
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L26(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L26(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L26:
	.long	.L41-.L26
	.long	.L40-.L26
	.long	.L39-.L26
	.long	.L38-.L26
	.long	.L37-.L26
	.long	.L36-.L26
	.long	.L35-.L26
	.long	.L34-.L26
	.long	.L33-.L26
	.long	.L32-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L24-.L26
	.long	.L31-.L26
	.long	.L30-.L26
	.long	.L29-.L26
	.long	.L28-.L26
	.long	.L27-.L26
	.long	.L25-.L26
	.text
.L25:
	movq	$12, -8(%rbp)
	jmp	.L42
.L27:
	movq	$2, -8(%rbp)
	jmp	.L42
.L28:
	movq	$13, -8(%rbp)
	jmp	.L42
.L29:
	movq	$18, -8(%rbp)
	jmp	.L42
.L30:
	movq	$1, -8(%rbp)
	jmp	.L42
.L31:
	movq	$3, -8(%rbp)
	jmp	.L42
.L32:
	movq	$5, -8(%rbp)
	jmp	.L42
.L33:
	movq	$6, -8(%rbp)
	jmp	.L42
.L34:
	movq	$9, -8(%rbp)
	jmp	.L42
.L35:
	movq	$4, -8(%rbp)
	jmp	.L42
.L36:
	movq	$0, -8(%rbp)
	jmp	.L42
.L37:
	movq	$11, -8(%rbp)
	jmp	.L42
.L38:
	movq	$17, -8(%rbp)
	jmp	.L42
.L39:
	movq	$8, -8(%rbp)
	jmp	.L42
.L40:
	movq	$14, -8(%rbp)
	jmp	.L42
.L41:
	movq	$7, -8(%rbp)
	jmp	.L42
.L24:
	movq	$10, -8(%rbp)
	nop
.L42:
	jmp	.L43
.L10:
	movl	$15, %eax
	jmp	.L23
.L14:
	movl	$2, %eax
	jmp	.L23
.L21:
	movl	$11, %eax
	jmp	.L23
.L19:
	movl	$10, %eax
	jmp	.L23
.L6:
	movl	$0, %eax
	jmp	.L23
.L11:
	movl	$4, %eax
	jmp	.L23
.L13:
	movl	$7, %eax
	jmp	.L23
.L9:
	movl	$13, %eax
	jmp	.L23
.L5:
	movl	$3, %eax
	jmp	.L23
.L16:
	movl	$8, %eax
	jmp	.L23
.L17:
	movl	$9, %eax
	jmp	.L23
.L12:
	movq	$16, -8(%rbp)
	jmp	.L43
.L22:
	movl	$5, %eax
	jmp	.L23
.L15:
	movl	$0, %eax
	jmp	.L23
.L20:
	movl	$14, %eax
	jmp	.L23
.L45:
	nop
.L43:
	jmp	.L44
.L23:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	char_to_int, .-char_to_int
	.globl	interpretation10_to
	.type	interpretation10_to, @function
interpretation10_to:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movl	%esi, -56(%rbp)
	movq	$8, -16(%rbp)
.L69:
	cmpq	$18, -16(%rbp)
	ja	.L70
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L49(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L49(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L49:
	.long	.L70-.L49
	.long	.L60-.L49
	.long	.L59-.L49
	.long	.L58-.L49
	.long	.L57-.L49
	.long	.L70-.L49
	.long	.L56-.L49
	.long	.L70-.L49
	.long	.L55-.L49
	.long	.L54-.L49
	.long	.L70-.L49
	.long	.L53-.L49
	.long	.L70-.L49
	.long	.L52-.L49
	.long	.L51-.L49
	.long	.L70-.L49
	.long	.L71-.L49
	.long	.L70-.L49
	.long	.L48-.L49
	.text
.L48:
	cmpl	$0, -36(%rbp)
	je	.L61
	movq	$1, -16(%rbp)
	jmp	.L63
.L61:
	movq	$3, -16(%rbp)
	jmp	.L63
.L57:
	movl	$0, -40(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, -36(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L63
.L51:
	movl	$18, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L63
.L55:
	movq	$14, -16(%rbp)
	jmp	.L63
.L60:
	movl	-36(%rbp), %eax
	movl	$0, %edx
	divl	-56(%rbp)
	movl	%edx, %eax
	movl	%eax, %edi
	call	int_to_char
	movl	%eax, -28(%rbp)
	movl	-40(%rbp), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	-28(%rbp), %edx
	movb	%dl, (%rax)
	movl	-36(%rbp), %eax
	movl	$0, %edx
	divl	-56(%rbp)
	movl	%eax, -36(%rbp)
	addl	$1, -40(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L63
.L58:
	movl	-40(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -32(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L63
.L53:
	cmpq	$0, -24(%rbp)
	jne	.L65
	movq	$6, -16(%rbp)
	jmp	.L63
.L65:
	movq	$4, -16(%rbp)
	jmp	.L63
.L54:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -16(%rbp)
	jmp	.L63
.L52:
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	subl	$1, -32(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L63
.L56:
	movl	$1, %edi
	call	exit@PLT
.L59:
	cmpl	$0, -32(%rbp)
	js	.L67
	movq	$13, -16(%rbp)
	jmp	.L63
.L67:
	movq	$9, -16(%rbp)
	jmp	.L63
.L70:
	nop
.L63:
	jmp	.L69
.L71:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	interpretation10_to, .-interpretation10_to
	.globl	int_to_char
	.type	int_to_char, @function
int_to_char:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$11, -8(%rbp)
.L115:
	cmpq	$18, -8(%rbp)
	ja	.L116
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L75(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L75(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L75:
	.long	.L93-.L75
	.long	.L92-.L75
	.long	.L91-.L75
	.long	.L90-.L75
	.long	.L89-.L75
	.long	.L88-.L75
	.long	.L87-.L75
	.long	.L86-.L75
	.long	.L85-.L75
	.long	.L84-.L75
	.long	.L83-.L75
	.long	.L82-.L75
	.long	.L81-.L75
	.long	.L80-.L75
	.long	.L79-.L75
	.long	.L78-.L75
	.long	.L77-.L75
	.long	.L76-.L75
	.long	.L74-.L75
	.text
.L74:
	movl	$50, %eax
	jmp	.L94
.L89:
	movl	$49, %eax
	jmp	.L94
.L79:
	movl	$69, %eax
	jmp	.L94
.L78:
	movl	$56, %eax
	jmp	.L94
.L81:
	movl	$55, %eax
	jmp	.L94
.L85:
	movl	$65, %eax
	jmp	.L94
.L92:
	movl	$54, %eax
	jmp	.L94
.L90:
	movl	$68, %eax
	jmp	.L94
.L77:
	movl	$48, %eax
	jmp	.L94
.L82:
	cmpl	$15, -20(%rbp)
	ja	.L95
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L97(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L97(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L97:
	.long	.L112-.L97
	.long	.L111-.L97
	.long	.L110-.L97
	.long	.L109-.L97
	.long	.L108-.L97
	.long	.L107-.L97
	.long	.L106-.L97
	.long	.L105-.L97
	.long	.L104-.L97
	.long	.L103-.L97
	.long	.L102-.L97
	.long	.L101-.L97
	.long	.L100-.L97
	.long	.L99-.L97
	.long	.L98-.L97
	.long	.L96-.L97
	.text
.L96:
	movq	$13, -8(%rbp)
	jmp	.L113
.L98:
	movq	$14, -8(%rbp)
	jmp	.L113
.L99:
	movq	$3, -8(%rbp)
	jmp	.L113
.L100:
	movq	$7, -8(%rbp)
	jmp	.L113
.L101:
	movq	$0, -8(%rbp)
	jmp	.L113
.L102:
	movq	$8, -8(%rbp)
	jmp	.L113
.L103:
	movq	$10, -8(%rbp)
	jmp	.L113
.L104:
	movq	$15, -8(%rbp)
	jmp	.L113
.L105:
	movq	$12, -8(%rbp)
	jmp	.L113
.L106:
	movq	$1, -8(%rbp)
	jmp	.L113
.L107:
	movq	$5, -8(%rbp)
	jmp	.L113
.L108:
	movq	$2, -8(%rbp)
	jmp	.L113
.L109:
	movq	$17, -8(%rbp)
	jmp	.L113
.L110:
	movq	$18, -8(%rbp)
	jmp	.L113
.L111:
	movq	$4, -8(%rbp)
	jmp	.L113
.L112:
	movq	$16, -8(%rbp)
	jmp	.L113
.L95:
	movq	$9, -8(%rbp)
	nop
.L113:
	jmp	.L114
.L84:
	movq	$6, -8(%rbp)
	jmp	.L114
.L80:
	movl	$70, %eax
	jmp	.L94
.L76:
	movl	$51, %eax
	jmp	.L94
.L87:
	movl	$0, %eax
	jmp	.L94
.L88:
	movl	$53, %eax
	jmp	.L94
.L83:
	movl	$57, %eax
	jmp	.L94
.L93:
	movl	$66, %eax
	jmp	.L94
.L86:
	movl	$67, %eax
	jmp	.L94
.L91:
	movl	$52, %eax
	jmp	.L94
.L116:
	nop
.L114:
	jmp	.L115
.L94:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	int_to_char, .-int_to_char
	.section	.rodata
.LC0:
	.string	"%d %d"
	.text
	.globl	interpretation
	.type	interpretation, @function
interpretation:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -24(%rbp)
.L141:
	cmpq	$20, -24(%rbp)
	ja	.L144
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L120(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L120(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L120:
	.long	.L132-.L120
	.long	.L131-.L120
	.long	.L144-.L120
	.long	.L130-.L120
	.long	.L144-.L120
	.long	.L144-.L120
	.long	.L144-.L120
	.long	.L129-.L120
	.long	.L128-.L120
	.long	.L144-.L120
	.long	.L127-.L120
	.long	.L126-.L120
	.long	.L125-.L120
	.long	.L124-.L120
	.long	.L123-.L120
	.long	.L122-.L120
	.long	.L144-.L120
	.long	.L144-.L120
	.long	.L144-.L120
	.long	.L121-.L120
	.long	.L145-.L120
	.text
.L123:
	leaq	-56(%rbp), %rdx
	leaq	-60(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -44(%rbp)
	movl	$0, -40(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L133
.L122:
	movl	-40(%rbp), %eax
	movslq	%eax, %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	char_to_int
	movl	%eax, -36(%rbp)
	movl	-60(%rbp), %eax
	imull	-44(%rbp), %eax
	movl	%eax, %edx
	movl	-36(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -44(%rbp)
	addl	$1, -40(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L133
.L125:
	movl	$18, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L133
.L128:
	movq	$12, -24(%rbp)
	jmp	.L133
.L131:
	movl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	-48(%rbp), %edx
	movb	%dl, (%rax)
	addl	$1, -52(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L133
.L130:
	cmpl	$32, -48(%rbp)
	jne	.L134
	movq	$14, -24(%rbp)
	jmp	.L133
.L134:
	movq	$1, -24(%rbp)
	jmp	.L133
.L126:
	cmpq	$0, -32(%rbp)
	jne	.L136
	movq	$19, -24(%rbp)
	jmp	.L133
.L136:
	movq	$10, -24(%rbp)
	jmp	.L133
.L124:
	movl	-40(%rbp), %eax
	cmpl	-52(%rbp), %eax
	jge	.L138
	movq	$15, -24(%rbp)
	jmp	.L133
.L138:
	movq	$0, -24(%rbp)
	jmp	.L133
.L121:
	movl	$1, %edi
	call	exit@PLT
.L127:
	movl	$0, -52(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L133
.L132:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	-56(%rbp), %edx
	movl	-44(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	interpretation10_to
	movq	$20, -24(%rbp)
	jmp	.L133
.L129:
	call	getchar@PLT
	movl	%eax, -48(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L133
.L144:
	nop
.L133:
	jmp	.L141
.L145:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L143
	call	__stack_chk_fail@PLT
.L143:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	interpretation, .-interpretation
	.globl	main
	.type	main, @function
main:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_kZqg_envp(%rip)
	nop
.L147:
	movq	$0, _TIG_IZ_kZqg_argv(%rip)
	nop
.L148:
	movl	$0, _TIG_IZ_kZqg_argc(%rip)
	nop
	nop
.L149:
.L150:
#APP
# 87 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-kZqg--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_kZqg_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_kZqg_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_kZqg_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L155:
	cmpq	$0, -8(%rbp)
	je	.L151
	cmpq	$1, -8(%rbp)
	jne	.L157
	movl	$0, %eax
	jmp	.L156
.L151:
	call	interpretation
	movq	$1, -8(%rbp)
	jmp	.L154
.L157:
	nop
.L154:
	jmp	.L155
.L156:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
