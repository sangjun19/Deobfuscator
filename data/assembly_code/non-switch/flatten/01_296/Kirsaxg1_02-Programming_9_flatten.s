	.file	"Kirsaxg1_02-Programming_9_flatten.c"
	.text
	.globl	_TIG_IZ_VUiJ_envp
	.bss
	.align 8
	.type	_TIG_IZ_VUiJ_envp, @object
	.size	_TIG_IZ_VUiJ_envp, 8
_TIG_IZ_VUiJ_envp:
	.zero	8
	.globl	_TIG_IZ_VUiJ_argv
	.align 8
	.type	_TIG_IZ_VUiJ_argv, @object
	.size	_TIG_IZ_VUiJ_argv, 8
_TIG_IZ_VUiJ_argv:
	.zero	8
	.globl	_TIG_IZ_VUiJ_argc
	.align 4
	.type	_TIG_IZ_VUiJ_argc, @object
	.size	_TIG_IZ_VUiJ_argc, 4
_TIG_IZ_VUiJ_argc:
	.zero	4
	.text
	.globl	print_memory_dump
	.type	print_memory_dump, @function
print_memory_dump:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -32(%rbp)
.L35:
	cmpq	$21, -32(%rbp)
	ja	.L38
	movq	-32(%rbp), %rax
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
	.long	.L39-.L4
	.long	.L38-.L4
	.long	.L38-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L38-.L4
	.long	.L39-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L38-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L39-.L4
	.long	.L38-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L18:
	cmpq	$0, -88(%rbp)
	jne	.L22
	movq	$0, -32(%rbp)
	jmp	.L24
.L22:
	movq	$12, -32(%rbp)
	jmp	.L24
.L9:
	movb	$0, -9(%rbp)
	leaq	-17(%rbp), %rdx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$8, -32(%rbp)
	jmp	.L24
.L11:
	cmpq	$0, -72(%rbp)
	jne	.L25
	movq	$18, -32(%rbp)
	jmp	.L24
.L25:
	movq	$7, -32(%rbp)
	jmp	.L24
.L14:
	movq	-80(%rbp), %rax
	subq	$1, %rax
	cmpq	%rax, -40(%rbp)
	jnb	.L27
	movq	$5, -32(%rbp)
	jmp	.L24
.L27:
	movq	$6, -32(%rbp)
	jmp	.L24
.L19:
	movl	$7, -52(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L24
.L8:
	movq	-48(%rbp), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %edx
	movl	-52(%rbp), %eax
	movl	%eax, %ecx
	sarl	%cl, %edx
	movl	%edx, %eax
	andl	$1, %eax
	testl	%eax, %eax
	je	.L29
	movq	$20, -32(%rbp)
	jmp	.L24
.L29:
	movq	$21, -32(%rbp)
	jmp	.L24
.L3:
	movl	$7, %eax
	subl	-52(%rbp), %eax
	cltq
	movb	$48, -17(%rbp,%rax)
	movq	$17, -32(%rbp)
	jmp	.L24
.L13:
	cmpl	$0, -52(%rbp)
	js	.L31
	movq	$16, -32(%rbp)
	jmp	.L24
.L31:
	movq	$14, -32(%rbp)
	jmp	.L24
.L10:
	movq	-40(%rbp), %rax
	cmpq	-80(%rbp), %rax
	jnb	.L33
	movq	$3, -32(%rbp)
	jmp	.L24
.L33:
	movq	$11, -32(%rbp)
	jmp	.L24
.L7:
	subl	$1, -52(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L24
.L16:
	addq	$1, -40(%rbp)
	movq	$13, -32(%rbp)
	jmp	.L24
.L17:
	movq	-88(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movw	$32, (%rax)
	movq	$6, -32(%rbp)
	jmp	.L24
.L15:
	movq	-72(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	-88(%rbp), %rax
	movb	$0, (%rax)
	movq	$0, -40(%rbp)
	movq	$13, -32(%rbp)
	jmp	.L24
.L5:
	movl	$7, %eax
	subl	-52(%rbp), %eax
	cltq
	movb	$49, -17(%rbp,%rax)
	movq	$17, -32(%rbp)
	jmp	.L24
.L38:
	nop
.L24:
	jmp	.L35
.L39:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L37
	call	__stack_chk_fail@PLT
.L37:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	print_memory_dump, .-print_memory_dump
	.globl	int_to_zeckendorf
	.type	int_to_zeckendorf, @function
int_to_zeckendorf:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$224, %rsp
	movl	%edi, -212(%rbp)
	movq	%rsi, -224(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$18, -200(%rbp)
.L76:
	cmpq	$26, -200(%rbp)
	ja	.L79
	movq	-200(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L43(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L43(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L43:
	.long	.L61-.L43
	.long	.L79-.L43
	.long	.L60-.L43
	.long	.L59-.L43
	.long	.L79-.L43
	.long	.L58-.L43
	.long	.L57-.L43
	.long	.L56-.L43
	.long	.L79-.L43
	.long	.L55-.L43
	.long	.L54-.L43
	.long	.L80-.L43
	.long	.L52-.L43
	.long	.L51-.L43
	.long	.L79-.L43
	.long	.L79-.L43
	.long	.L50-.L43
	.long	.L80-.L43
	.long	.L48-.L43
	.long	.L47-.L43
	.long	.L80-.L43
	.long	.L45-.L43
	.long	.L79-.L43
	.long	.L44-.L43
	.long	.L79-.L43
	.long	.L79-.L43
	.long	.L42-.L43
	.text
.L48:
	cmpq	$0, -224(%rbp)
	jne	.L62
	movq	$11, -200(%rbp)
	jmp	.L64
.L62:
	movq	$3, -200(%rbp)
	jmp	.L64
.L52:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-224(%rbp), %rax
	addq	%rdx, %rax
	movw	$48, (%rax)
	movq	$26, -200(%rbp)
	jmp	.L64
.L44:
	subl	$1, -204(%rbp)
	movq	$10, -200(%rbp)
	jmp	.L64
.L59:
	cmpl	$0, -212(%rbp)
	jne	.L65
	movq	$19, -200(%rbp)
	jmp	.L64
.L65:
	movq	$7, -200(%rbp)
	jmp	.L64
.L50:
	cmpl	$0, -204(%rbp)
	js	.L67
	movq	$13, -200(%rbp)
	jmp	.L64
.L67:
	movq	$0, -200(%rbp)
	jmp	.L64
.L45:
	cmpl	$45, -208(%rbp)
	jg	.L69
	movq	$9, -200(%rbp)
	jmp	.L64
.L69:
	movq	$6, -200(%rbp)
	jmp	.L64
.L42:
	subl	$1, -204(%rbp)
	movq	$16, -200(%rbp)
	jmp	.L64
.L55:
	movl	-208(%rbp), %eax
	subl	$1, %eax
	cltq
	movl	-192(%rbp,%rax,4), %edx
	movl	-208(%rbp), %eax
	subl	$2, %eax
	cltq
	movl	-192(%rbp,%rax,4), %eax
	addl	%eax, %edx
	movl	-208(%rbp), %eax
	cltq
	movl	%edx, -192(%rbp,%rax,4)
	addl	$1, -208(%rbp)
	movq	$21, -200(%rbp)
	jmp	.L64
.L51:
	movl	-204(%rbp), %eax
	cltq
	movl	-192(%rbp,%rax,4), %eax
	cmpl	%eax, -212(%rbp)
	jb	.L72
	movq	$2, -200(%rbp)
	jmp	.L64
.L72:
	movq	$12, -200(%rbp)
	jmp	.L64
.L47:
	movq	-224(%rbp), %rax
	movw	$48, (%rax)
	movq	$20, -200(%rbp)
	jmp	.L64
.L57:
	movl	$45, -204(%rbp)
	movq	$10, -200(%rbp)
	jmp	.L64
.L58:
	movq	-224(%rbp), %rax
	movb	$0, (%rax)
	movq	$16, -200(%rbp)
	jmp	.L64
.L54:
	movl	-204(%rbp), %eax
	cltq
	movl	-192(%rbp,%rax,4), %eax
	cmpl	%eax, -212(%rbp)
	jnb	.L74
	movq	$23, -200(%rbp)
	jmp	.L64
.L74:
	movq	$5, -200(%rbp)
	jmp	.L64
.L61:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-224(%rbp), %rax
	addq	%rdx, %rax
	movw	$49, (%rax)
	movq	$17, -200(%rbp)
	jmp	.L64
.L56:
	movl	$1, -192(%rbp)
	movl	$2, -188(%rbp)
	movl	$2, -208(%rbp)
	movq	$21, -200(%rbp)
	jmp	.L64
.L60:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-224(%rbp), %rax
	addq	%rdx, %rax
	movw	$49, (%rax)
	movl	-204(%rbp), %eax
	cltq
	movl	-192(%rbp,%rax,4), %eax
	subl	%eax, -212(%rbp)
	movq	$26, -200(%rbp)
	jmp	.L64
.L79:
	nop
.L64:
	jmp	.L76
.L80:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L78
	call	__stack_chk_fail@PLT
.L78:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	int_to_zeckendorf, .-int_to_zeckendorf
	.globl	int_to_base
	.type	int_to_base, @function
int_to_base:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movl	%esi, -88(%rbp)
	movq	%rdx, -96(%rbp)
	movl	%ecx, -100(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$22, -56(%rbp)
.L122:
	cmpq	$30, -56(%rbp)
	ja	.L125
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L84(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L84(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L84:
	.long	.L103-.L84
	.long	.L102-.L84
	.long	.L126-.L84
	.long	.L100-.L84
	.long	.L125-.L84
	.long	.L99-.L84
	.long	.L98-.L84
	.long	.L126-.L84
	.long	.L96-.L84
	.long	.L126-.L84
	.long	.L125-.L84
	.long	.L94-.L84
	.long	.L93-.L84
	.long	.L92-.L84
	.long	.L91-.L84
	.long	.L125-.L84
	.long	.L125-.L84
	.long	.L126-.L84
	.long	.L125-.L84
	.long	.L89-.L84
	.long	.L125-.L84
	.long	.L125-.L84
	.long	.L88-.L84
	.long	.L87-.L84
	.long	.L125-.L84
	.long	.L125-.L84
	.long	.L125-.L84
	.long	.L86-.L84
	.long	.L125-.L84
	.long	.L85-.L84
	.long	.L83-.L84
	.text
.L83:
	cmpl	$36, -88(%rbp)
	jle	.L104
	movq	$9, -56(%rbp)
	jmp	.L106
.L104:
	movq	$0, -56(%rbp)
	jmp	.L106
.L91:
	movl	-76(%rbp), %eax
	movl	%eax, -64(%rbp)
	addl	$1, -76(%rbp)
	movl	-84(%rbp), %eax
	cltd
	idivl	-88(%rbp)
	movl	%edx, %ecx
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movzbl	-48(%rbp,%rax), %eax
	movb	%al, (%rdx)
	movl	-84(%rbp), %eax
	cltd
	idivl	-88(%rbp)
	movl	%eax, -84(%rbp)
	movq	$29, -56(%rbp)
	jmp	.L106
.L93:
	cmpl	$0, -72(%rbp)
	je	.L107
	movq	$1, -56(%rbp)
	jmp	.L106
.L107:
	movq	$3, -56(%rbp)
	jmp	.L106
.L96:
	movb	$65, -38(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L106
.L102:
	movl	-76(%rbp), %eax
	movl	%eax, -60(%rbp)
	addl	$1, -76(%rbp)
	movl	-60(%rbp), %eax
	movslq	%eax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movb	$45, (%rax)
	movq	$3, -56(%rbp)
	jmp	.L106
.L87:
	movl	$1, -72(%rbp)
	negl	-84(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L106
.L100:
	movl	-76(%rbp), %eax
	movslq	%eax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movl	$0, -68(%rbp)
	movq	$5, -56(%rbp)
	jmp	.L106
.L94:
	movl	$0, -76(%rbp)
	movl	$0, -72(%rbp)
	movq	$6, -56(%rbp)
	jmp	.L106
.L92:
	cmpl	$1, -88(%rbp)
	jg	.L110
	movq	$17, -56(%rbp)
	jmp	.L106
.L110:
	movq	$30, -56(%rbp)
	jmp	.L106
.L89:
	cmpl	$0, -100(%rbp)
	je	.L112
	movq	$8, -56(%rbp)
	jmp	.L106
.L112:
	movq	$11, -56(%rbp)
	jmp	.L106
.L98:
	cmpl	$0, -84(%rbp)
	jns	.L114
	movq	$23, -56(%rbp)
	jmp	.L106
.L114:
	movq	$14, -56(%rbp)
	jmp	.L106
.L86:
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -77(%rbp)
	movl	-76(%rbp), %eax
	subl	-68(%rbp), %eax
	cltq
	leaq	-1(%rax), %rdx
	movq	-96(%rbp), %rax
	addq	%rdx, %rax
	movl	-68(%rbp), %edx
	movslq	%edx, %rcx
	movq	-96(%rbp), %rdx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	movl	-76(%rbp), %eax
	subl	-68(%rbp), %eax
	cltq
	leaq	-1(%rax), %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-77(%rbp), %eax
	movb	%al, (%rdx)
	addl	$1, -68(%rbp)
	movq	$5, -56(%rbp)
	jmp	.L106
.L88:
	cmpq	$0, -96(%rbp)
	jne	.L116
	movq	$7, -56(%rbp)
	jmp	.L106
.L116:
	movq	$13, -56(%rbp)
	jmp	.L106
.L99:
	movl	-76(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cmpl	%eax, -68(%rbp)
	jge	.L118
	movq	$27, -56(%rbp)
	jmp	.L106
.L118:
	movq	$2, -56(%rbp)
	jmp	.L106
.L103:
	movb	$48, -48(%rbp)
	movb	$49, -47(%rbp)
	movb	$50, -46(%rbp)
	movb	$51, -45(%rbp)
	movb	$52, -44(%rbp)
	movb	$53, -43(%rbp)
	movb	$54, -42(%rbp)
	movb	$55, -41(%rbp)
	movb	$56, -40(%rbp)
	movb	$57, -39(%rbp)
	movb	$97, -38(%rbp)
	movb	$98, -37(%rbp)
	movb	$99, -36(%rbp)
	movb	$100, -35(%rbp)
	movb	$101, -34(%rbp)
	movb	$102, -33(%rbp)
	movb	$103, -32(%rbp)
	movb	$104, -31(%rbp)
	movb	$105, -30(%rbp)
	movb	$106, -29(%rbp)
	movb	$107, -28(%rbp)
	movb	$108, -27(%rbp)
	movb	$109, -26(%rbp)
	movb	$110, -25(%rbp)
	movb	$111, -24(%rbp)
	movb	$112, -23(%rbp)
	movb	$113, -22(%rbp)
	movb	$114, -21(%rbp)
	movb	$115, -20(%rbp)
	movb	$116, -19(%rbp)
	movb	$117, -18(%rbp)
	movb	$118, -17(%rbp)
	movb	$119, -16(%rbp)
	movb	$120, -15(%rbp)
	movb	$121, -14(%rbp)
	movb	$122, -13(%rbp)
	movb	$0, -12(%rbp)
	movq	$19, -56(%rbp)
	jmp	.L106
.L85:
	cmpl	$0, -84(%rbp)
	jle	.L120
	movq	$14, -56(%rbp)
	jmp	.L106
.L120:
	movq	$12, -56(%rbp)
	jmp	.L106
.L125:
	nop
.L106:
	jmp	.L122
.L126:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L124
	call	__stack_chk_fail@PLT
.L124:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	int_to_base, .-int_to_base
	.section	.rodata
.LC0:
	.string	"%s"
.LC1:
	.string	"%d"
	.text
	.globl	overprintf
	.type	overprintf, @function
overprintf:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1280, %rsp
	movq	%rdi, -1272(%rbp)
	movq	%rsi, -168(%rbp)
	movq	%rdx, -160(%rbp)
	movq	%rcx, -152(%rbp)
	movq	%r8, -144(%rbp)
	movq	%r9, -136(%rbp)
	testb	%al, %al
	je	.L128
	movaps	%xmm0, -128(%rbp)
	movaps	%xmm1, -112(%rbp)
	movaps	%xmm2, -96(%rbp)
	movaps	%xmm3, -80(%rbp)
	movaps	%xmm4, -64(%rbp)
	movaps	%xmm5, -48(%rbp)
	movaps	%xmm6, -32(%rbp)
	movaps	%xmm7, -16(%rbp)
.L128:
	movq	%fs:40, %rax
	movq	%rax, -184(%rbp)
	xorl	%eax, %eax
	movq	$75, -1072(%rbp)
.L288:
	movq	-1072(%rbp), %rax
	subq	$3, %rax
	cmpq	$115, %rax
	ja	.L291
	leaq	0(,%rax,4), %rdx
	leaq	.L131(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L131(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L131:
	.long	.L195-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L194-.L131
	.long	.L193-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L192-.L131
	.long	.L291-.L131
	.long	.L191-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L190-.L131
	.long	.L189-.L131
	.long	.L291-.L131
	.long	.L188-.L131
	.long	.L187-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L186-.L131
	.long	.L291-.L131
	.long	.L185-.L131
	.long	.L291-.L131
	.long	.L184-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L183-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L182-.L131
	.long	.L181-.L131
	.long	.L291-.L131
	.long	.L180-.L131
	.long	.L291-.L131
	.long	.L179-.L131
	.long	.L178-.L131
	.long	.L291-.L131
	.long	.L177-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L176-.L131
	.long	.L291-.L131
	.long	.L175-.L131
	.long	.L174-.L131
	.long	.L173-.L131
	.long	.L291-.L131
	.long	.L172-.L131
	.long	.L171-.L131
	.long	.L291-.L131
	.long	.L170-.L131
	.long	.L169-.L131
	.long	.L168-.L131
	.long	.L167-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L166-.L131
	.long	.L165-.L131
	.long	.L291-.L131
	.long	.L164-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L163-.L131
	.long	.L162-.L131
	.long	.L161-.L131
	.long	.L160-.L131
	.long	.L159-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L158-.L131
	.long	.L291-.L131
	.long	.L157-.L131
	.long	.L156-.L131
	.long	.L291-.L131
	.long	.L155-.L131
	.long	.L154-.L131
	.long	.L153-.L131
	.long	.L291-.L131
	.long	.L152-.L131
	.long	.L291-.L131
	.long	.L151-.L131
	.long	.L150-.L131
	.long	.L149-.L131
	.long	.L148-.L131
	.long	.L147-.L131
	.long	.L146-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L145-.L131
	.long	.L144-.L131
	.long	.L143-.L131
	.long	.L291-.L131
	.long	.L142-.L131
	.long	.L141-.L131
	.long	.L291-.L131
	.long	.L140-.L131
	.long	.L291-.L131
	.long	.L139-.L131
	.long	.L138-.L131
	.long	.L137-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L136-.L131
	.long	.L291-.L131
	.long	.L135-.L131
	.long	.L134-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L133-.L131
	.long	.L291-.L131
	.long	.L132-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L291-.L131
	.long	.L130-.L131
	.text
.L188:
	movl	-1236(%rbp), %eax
	movb	$0, -992(%rbp,%rax)
	addl	$1, -1236(%rbp)
	movq	$59, -1072(%rbp)
	jmp	.L196
.L172:
	movq	$15, -1072(%rbp)
	jmp	.L196
.L134:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$109, %al
	jne	.L197
	movq	$47, -1072(%rbp)
	jmp	.L196
.L197:
	movq	$46, -1072(%rbp)
	jmp	.L196
.L164:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$37, %al
	jne	.L199
	movq	$74, -1072(%rbp)
	jmp	.L196
.L199:
	movq	$41, -1072(%rbp)
	jmp	.L196
.L137:
	movl	-1200(%rbp), %eax
	movb	$0, -624(%rbp,%rax)
	addl	$1, -1200(%rbp)
	movq	$92, -1072(%rbp)
	jmp	.L196
.L190:
	movl	-1244(%rbp), %eax
	jmp	.L289
.L167:
	leaq	-400(%rbp), %rdx
	leaq	-1080(%rbp), %rax
	movl	$8, %esi
	movq	%rax, %rdi
	call	print_memory_dump
	leaq	-400(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1156(%rbp)
	movl	-1156(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L153:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$90, %al
	jne	.L202
	movq	$78, -1072(%rbp)
	jmp	.L196
.L202:
	movq	$60, -1072(%rbp)
	jmp	.L196
.L191:
	leaq	-736(%rbp), %rdx
	movl	-1208(%rbp), %esi
	movl	-1212(%rbp), %eax
	movl	$1, %ecx
	movl	%eax, %edi
	call	int_to_base
	leaq	-736(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1088(%rbp)
	movl	-1088(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L138:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$84, %al
	jne	.L204
	movq	$96, -1072(%rbp)
	jmp	.L196
.L204:
	movq	$108, -1072(%rbp)
	jmp	.L196
.L159:
	cmpl	$99, -1188(%rbp)
	jbe	.L206
	movq	$91, -1072(%rbp)
	jmp	.L196
.L206:
	movq	$86, -1072(%rbp)
	jmp	.L196
.L141:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$79, %al
	jne	.L208
	movq	$33, -1072(%rbp)
	jmp	.L196
.L208:
	movq	$108, -1072(%rbp)
	jmp	.L196
.L169:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$100, %al
	jne	.L210
	movq	$98, -1072(%rbp)
	jmp	.L196
.L210:
	movq	$38, -1072(%rbp)
	jmp	.L196
.L154:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$114, %al
	jne	.L212
	movq	$39, -1072(%rbp)
	jmp	.L196
.L212:
	movq	$60, -1072(%rbp)
	jmp	.L196
.L130:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L214
	movq	$62, -1072(%rbp)
	jmp	.L196
.L214:
	movq	$50, -1072(%rbp)
	jmp	.L196
.L152:
	cmpl	$99, -1228(%rbp)
	jbe	.L216
	movq	$24, -1072(%rbp)
	jmp	.L196
.L216:
	movq	$51, -1072(%rbp)
	jmp	.L196
.L155:
	addq	$2, -1272(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L218
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L219
.L218:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L219:
	movl	(%rax), %eax
	movl	%eax, -1180(%rbp)
	movl	-1180(%rbp), %eax
	movl	%eax, -1224(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L220
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L221
.L220:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L221:
	movl	(%rax), %eax
	movl	%eax, -1176(%rbp)
	movl	-1176(%rbp), %eax
	movl	%eax, -1220(%rbp)
	movb	$0, -848(%rbp)
	movl	$1, -1216(%rbp)
	movq	$112, -1072(%rbp)
	jmp	.L196
.L195:
	leaq	-512(%rbp), %rdx
	leaq	-1252(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	print_memory_dump
	leaq	-512(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1084(%rbp)
	movl	-1084(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L189:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$111, %al
	jne	.L222
	movq	$36, -1072(%rbp)
	jmp	.L196
.L222:
	movq	$79, -1072(%rbp)
	jmp	.L196
.L185:
	leaq	-960(%rbp), %rdx
	movl	-1232(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	int_to_zeckendorf
	leaq	-960(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1092(%rbp)
	movl	-1092(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L180:
	addq	$2, -1272(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L224
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L225
.L224:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L225:
	movl	(%rax), %eax
	movl	%eax, -1120(%rbp)
	movl	-1120(%rbp), %eax
	movl	%eax, -1240(%rbp)
	movb	$0, -992(%rbp)
	movl	$1, -1236(%rbp)
	movq	$59, -1072(%rbp)
	jmp	.L196
.L160:
	leaq	-624(%rbp), %rdx
	leaq	-1256(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	print_memory_dump
	leaq	-624(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1164(%rbp)
	movl	-1164(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L149:
	movl	-1216(%rbp), %eax
	movb	$0, -848(%rbp,%rax)
	addl	$1, -1216(%rbp)
	movq	$112, -1072(%rbp)
	jmp	.L196
.L139:
	movl	$8, -1016(%rbp)
	movl	$48, -1012(%rbp)
	leaq	16(%rbp), %rax
	movq	%rax, -1008(%rbp)
	leaq	-176(%rbp), %rax
	movq	%rax, -1000(%rbp)
	movl	$0, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L184:
	movl	$-1, %eax
	jmp	.L289
.L140:
	addq	$2, -1272(%rbp)
	movl	-1012(%rbp), %eax
	cmpl	$175, %eax
	ja	.L226
	movq	-1000(%rbp), %rax
	movl	-1012(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1012(%rbp), %edx
	addl	$16, %edx
	movl	%edx, -1012(%rbp)
	jmp	.L227
.L226:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L227:
	movsd	(%rax), %xmm0
	movsd	%xmm0, -1056(%rbp)
	movsd	-1056(%rbp), %xmm0
	movsd	%xmm0, -1080(%rbp)
	movb	$0, -400(%rbp)
	movl	$1, -1192(%rbp)
	movq	$55, -1072(%rbp)
	jmp	.L196
.L136:
	addq	$2, -1272(%rbp)
	movl	-1012(%rbp), %eax
	cmpl	$175, %eax
	ja	.L228
	movq	-1000(%rbp), %rax
	movl	-1012(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1012(%rbp), %edx
	addl	$16, %edx
	movl	%edx, -1012(%rbp)
	jmp	.L229
.L228:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L229:
	movsd	(%rax), %xmm0
	movsd	%xmm0, -1064(%rbp)
	pxor	%xmm0, %xmm0
	cvtsd2ss	-1064(%rbp), %xmm0
	movss	%xmm0, -1248(%rbp)
	movb	$0, -288(%rbp)
	movl	$1, -1188(%rbp)
	movq	$69, -1072(%rbp)
	jmp	.L196
.L171:
	movl	-1228(%rbp), %eax
	movb	$0, -960(%rbp,%rax)
	addl	$1, -1228(%rbp)
	movq	$81, -1072(%rbp)
	jmp	.L196
.L135:
	movl	-1204(%rbp), %eax
	movb	$0, -736(%rbp,%rax)
	addl	$1, -1204(%rbp)
	movq	$87, -1072(%rbp)
	jmp	.L196
.L187:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$86, %al
	jne	.L230
	movq	$72, -1072(%rbp)
	jmp	.L196
.L230:
	movq	$65, -1072(%rbp)
	jmp	.L196
.L161:
	leaq	-848(%rbp), %rdx
	movl	-1220(%rbp), %esi
	movl	-1224(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	int_to_base
	leaq	-848(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1160(%rbp)
	movl	-1160(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L168:
	cmpl	$99, -1192(%rbp)
	jbe	.L232
	movq	$56, -1072(%rbp)
	jmp	.L196
.L232:
	movq	$114, -1072(%rbp)
	jmp	.L196
.L165:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$67, %al
	jne	.L234
	movq	$84, -1072(%rbp)
	jmp	.L196
.L234:
	movq	$66, -1072(%rbp)
	jmp	.L196
.L166:
	cmpl	$19, -1236(%rbp)
	jbe	.L236
	movq	$44, -1072(%rbp)
	jmp	.L196
.L236:
	movq	$18, -1072(%rbp)
	jmp	.L196
.L194:
	cmpl	$99, -1196(%rbp)
	jbe	.L238
	movq	$3, -1072(%rbp)
	jmp	.L196
.L238:
	movq	$22, -1072(%rbp)
	jmp	.L196
.L179:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$109, %al
	jne	.L240
	movq	$10, -1072(%rbp)
	jmp	.L196
.L240:
	movq	$95, -1072(%rbp)
	jmp	.L196
.L147:
	cmpl	$99, -1204(%rbp)
	jbe	.L242
	movq	$12, -1072(%rbp)
	jmp	.L196
.L242:
	movq	$107, -1072(%rbp)
	jmp	.L196
.L150:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$118, %al
	jne	.L244
	movq	$77, -1072(%rbp)
	jmp	.L196
.L244:
	movq	$66, -1072(%rbp)
	jmp	.L196
.L181:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$109, %al
	jne	.L246
	movq	$54, -1072(%rbp)
	jmp	.L196
.L246:
	movq	$38, -1072(%rbp)
	jmp	.L196
.L157:
	addq	$1, -1272(%rbp)
	movq	$83, -1072(%rbp)
	jmp	.L196
.L133:
	cmpl	$99, -1216(%rbp)
	jbe	.L248
	movq	$67, -1072(%rbp)
	jmp	.L196
.L248:
	movq	$85, -1072(%rbp)
	jmp	.L196
.L156:
	cmpq	$0, -1272(%rbp)
	jne	.L250
	movq	$26, -1072(%rbp)
	jmp	.L196
.L250:
	movq	$100, -1072(%rbp)
	jmp	.L196
.L173:
	addq	$2, -1272(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L252
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L253
.L252:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L253:
	movq	(%rax), %rax
	movq	%rax, -1048(%rbp)
	movq	-1048(%rbp), %rax
	movq	%rax, -1040(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L254
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L255
.L254:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L255:
	movl	(%rax), %eax
	movl	%eax, -1148(%rbp)
	movl	-1148(%rbp), %eax
	movl	%eax, -1144(%rbp)
	movl	-1144(%rbp), %edx
	movq	-1040(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	str_to_int
	movl	%eax, -1140(%rbp)
	movl	-1140(%rbp), %eax
	movl	%eax, -1136(%rbp)
	movl	-1136(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1132(%rbp)
	movl	-1132(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L186:
	movl	-1196(%rbp), %eax
	movb	$0, -512(%rbp,%rax)
	addl	$1, -1196(%rbp)
	movq	$6, -1072(%rbp)
	jmp	.L196
.L170:
	addq	$2, -1272(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L256
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L257
.L256:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L257:
	movl	(%rax), %eax
	movl	%eax, -1152(%rbp)
	movl	-1152(%rbp), %eax
	movl	%eax, -1252(%rbp)
	movb	$0, -512(%rbp)
	movl	$1, -1196(%rbp)
	movq	$6, -1072(%rbp)
	jmp	.L196
.L163:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$116, %al
	jne	.L258
	movq	$93, -1072(%rbp)
	jmp	.L196
.L258:
	movq	$101, -1072(%rbp)
	jmp	.L196
.L174:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$105, %al
	jne	.L260
	movq	$29, -1072(%rbp)
	jmp	.L196
.L260:
	movq	$46, -1072(%rbp)
	jmp	.L196
.L176:
	leaq	-992(%rbp), %rdx
	movl	-1240(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	int_to_roman
	leaq	-992(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1128(%rbp)
	movl	-1128(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L145:
	leaq	-288(%rbp), %rdx
	leaq	-1248(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	print_memory_dump
	leaq	-288(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1184(%rbp)
	movl	-1184(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L158:
	addq	$2, -1272(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L262
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L263
.L262:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L263:
	movl	(%rax), %eax
	movl	%eax, -1172(%rbp)
	movl	-1172(%rbp), %eax
	movl	%eax, -1212(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L264
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L265
.L264:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L265:
	movl	(%rax), %eax
	movl	%eax, -1168(%rbp)
	movl	-1168(%rbp), %eax
	movl	%eax, -1208(%rbp)
	movb	$0, -736(%rbp)
	movl	$1, -1204(%rbp)
	movq	$87, -1072(%rbp)
	jmp	.L196
.L132:
	movl	-1192(%rbp), %eax
	movb	$0, -400(%rbp,%rax)
	addl	$1, -1192(%rbp)
	movq	$55, -1072(%rbp)
	jmp	.L196
.L182:
	addq	$2, -1272(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L266
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L267
.L266:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L267:
	movq	(%rax), %rax
	movq	%rax, -1032(%rbp)
	movq	-1032(%rbp), %rax
	movq	%rax, -1024(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L268
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L269
.L268:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L269:
	movl	(%rax), %eax
	movl	%eax, -1116(%rbp)
	movl	-1116(%rbp), %eax
	movl	%eax, -1112(%rbp)
	movl	-1112(%rbp), %edx
	movq	-1024(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	str_to_int
	movl	%eax, -1108(%rbp)
	movl	-1108(%rbp), %eax
	movl	%eax, -1104(%rbp)
	movl	-1104(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	%eax, -1100(%rbp)
	movl	-1100(%rbp), %eax
	addl	%eax, -1244(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L143:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$111, %al
	jne	.L270
	movq	$48, -1072(%rbp)
	jmp	.L196
.L270:
	movq	$101, -1072(%rbp)
	jmp	.L196
.L177:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -1244(%rbp)
	addq	$1, -1272(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L142:
	movl	$37, %edi
	call	putchar@PLT
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$2, -1244(%rbp)
	addq	$1, -1272(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L144:
	cmpl	$99, -1200(%rbp)
	jbe	.L272
	movq	$68, -1072(%rbp)
	jmp	.L196
.L272:
	movq	$102, -1072(%rbp)
	jmp	.L196
.L192:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$102, %al
	jne	.L274
	movq	$105, -1072(%rbp)
	jmp	.L196
.L274:
	movq	$7, -1072(%rbp)
	jmp	.L196
.L175:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$109, %al
	jne	.L276
	movq	$88, -1072(%rbp)
	jmp	.L196
.L276:
	movq	$34, -1072(%rbp)
	jmp	.L196
.L178:
	addq	$2, -1272(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L278
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L279
.L278:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L279:
	movl	(%rax), %eax
	movl	%eax, -1124(%rbp)
	movl	-1124(%rbp), %eax
	movl	%eax, -1232(%rbp)
	movb	$0, -960(%rbp)
	movl	$1, -1228(%rbp)
	movq	$81, -1072(%rbp)
	jmp	.L196
.L162:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$67, %al
	jne	.L280
	movq	$19, -1072(%rbp)
	jmp	.L196
.L280:
	movq	$65, -1072(%rbp)
	jmp	.L196
.L151:
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$82, %al
	jne	.L282
	movq	$16, -1072(%rbp)
	jmp	.L196
.L282:
	movq	$79, -1072(%rbp)
	jmp	.L196
.L193:
	movl	$37, %edi
	call	putchar@PLT
	movq	-1272(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$2, -1244(%rbp)
	addq	$1, -1272(%rbp)
	movq	$118, -1072(%rbp)
	jmp	.L196
.L146:
	movq	-1272(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$117, %al
	jne	.L284
	movq	$53, -1072(%rbp)
	jmp	.L196
.L284:
	movq	$34, -1072(%rbp)
	jmp	.L196
.L183:
	addq	$2, -1272(%rbp)
	movl	-1016(%rbp), %eax
	cmpl	$47, %eax
	ja	.L286
	movq	-1000(%rbp), %rax
	movl	-1016(%rbp), %edx
	movl	%edx, %edx
	addq	%rdx, %rax
	movl	-1016(%rbp), %edx
	addl	$8, %edx
	movl	%edx, -1016(%rbp)
	jmp	.L287
.L286:
	movq	-1008(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	%rdx, -1008(%rbp)
.L287:
	movl	(%rax), %eax
	movl	%eax, -1096(%rbp)
	movl	-1096(%rbp), %eax
	movl	%eax, -1256(%rbp)
	movb	$0, -624(%rbp)
	movl	$1, -1200(%rbp)
	movq	$92, -1072(%rbp)
	jmp	.L196
.L148:
	movl	-1188(%rbp), %eax
	movb	$0, -288(%rbp,%rax)
	addl	$1, -1188(%rbp)
	movq	$69, -1072(%rbp)
	jmp	.L196
.L291:
	nop
.L196:
	jmp	.L288
.L289:
	movq	-184(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L290
	call	__stack_chk_fail@PLT
.L290:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	overprintf, .-overprintf
	.section	.rodata
.LC2:
	.string	"M"
.LC3:
	.string	"CM"
.LC4:
	.string	"D"
.LC5:
	.string	"CD"
.LC6:
	.string	"C"
.LC7:
	.string	"XC"
.LC8:
	.string	"L"
.LC9:
	.string	"XL"
.LC10:
	.string	"X"
.LC11:
	.string	"IX"
.LC12:
	.string	"V"
.LC13:
	.string	"IV"
.LC14:
	.string	"I"
	.text
	.globl	int_to_roman
	.type	int_to_roman, @function
int_to_roman:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movl	%edi, -196(%rbp)
	movq	%rsi, -208(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -184(%rbp)
.L311:
	cmpq	$11, -184(%rbp)
	ja	.L314
	movq	-184(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L295(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L295(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L295:
	.long	.L302-.L295
	.long	.L301-.L295
	.long	.L314-.L295
	.long	.L300-.L295
	.long	.L314-.L295
	.long	.L314-.L295
	.long	.L299-.L295
	.long	.L315-.L295
	.long	.L315-.L295
	.long	.L296-.L295
	.long	.L314-.L295
	.long	.L294-.L295
	.text
.L301:
	cmpq	$0, -208(%rbp)
	jne	.L304
	movq	$7, -184(%rbp)
	jmp	.L306
.L304:
	movq	$11, -184(%rbp)
	jmp	.L306
.L300:
	movl	-188(%rbp), %eax
	cltq
	movq	-112(%rbp,%rax,8), %rdx
	movq	-208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movl	-188(%rbp), %eax
	cltq
	movl	-176(%rbp,%rax,4), %eax
	subl	%eax, -196(%rbp)
	movq	$6, -184(%rbp)
	jmp	.L306
.L294:
	movl	$1000, -176(%rbp)
	movl	$900, -172(%rbp)
	movl	$500, -168(%rbp)
	movl	$400, -164(%rbp)
	movl	$100, -160(%rbp)
	movl	$90, -156(%rbp)
	movl	$50, -152(%rbp)
	movl	$40, -148(%rbp)
	movl	$10, -144(%rbp)
	movl	$9, -140(%rbp)
	movl	$5, -136(%rbp)
	movl	$4, -132(%rbp)
	movl	$1, -128(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, -112(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, -104(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, -96(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, -88(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, -80(%rbp)
	leaq	.LC7(%rip), %rax
	movq	%rax, -72(%rbp)
	leaq	.LC8(%rip), %rax
	movq	%rax, -64(%rbp)
	leaq	.LC9(%rip), %rax
	movq	%rax, -56(%rbp)
	leaq	.LC10(%rip), %rax
	movq	%rax, -48(%rbp)
	leaq	.LC11(%rip), %rax
	movq	%rax, -40(%rbp)
	leaq	.LC12(%rip), %rax
	movq	%rax, -32(%rbp)
	leaq	.LC13(%rip), %rax
	movq	%rax, -24(%rbp)
	leaq	.LC14(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	-208(%rbp), %rax
	movb	$0, (%rax)
	movl	$0, -188(%rbp)
	movq	$0, -184(%rbp)
	jmp	.L306
.L296:
	addl	$1, -188(%rbp)
	movq	$0, -184(%rbp)
	jmp	.L306
.L299:
	movl	-188(%rbp), %eax
	cltq
	movl	-176(%rbp,%rax,4), %eax
	cmpl	%eax, -196(%rbp)
	jl	.L307
	movq	$3, -184(%rbp)
	jmp	.L306
.L307:
	movq	$9, -184(%rbp)
	jmp	.L306
.L302:
	cmpl	$12, -188(%rbp)
	jg	.L309
	movq	$6, -184(%rbp)
	jmp	.L306
.L309:
	movq	$8, -184(%rbp)
	jmp	.L306
.L314:
	nop
.L306:
	jmp	.L311
.L315:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L313
	call	__stack_chk_fail@PLT
.L313:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	int_to_roman, .-int_to_roman
	.globl	str_to_int
	.type	str_to_int, @function
str_to_int:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movq	$1, -8(%rbp)
.L358:
	cmpq	$27, -8(%rbp)
	ja	.L359
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L319(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L319(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L319:
	.long	.L339-.L319
	.long	.L338-.L319
	.long	.L359-.L319
	.long	.L337-.L319
	.long	.L359-.L319
	.long	.L336-.L319
	.long	.L335-.L319
	.long	.L334-.L319
	.long	.L333-.L319
	.long	.L332-.L319
	.long	.L359-.L319
	.long	.L331-.L319
	.long	.L359-.L319
	.long	.L330-.L319
	.long	.L359-.L319
	.long	.L359-.L319
	.long	.L329-.L319
	.long	.L328-.L319
	.long	.L327-.L319
	.long	.L359-.L319
	.long	.L326-.L319
	.long	.L325-.L319
	.long	.L324-.L319
	.long	.L323-.L319
	.long	.L322-.L319
	.long	.L321-.L319
	.long	.L320-.L319
	.long	.L318-.L319
	.text
.L327:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -45(%rbp)
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	movq	$24, -8(%rbp)
	jmp	.L340
.L321:
	movl	$0, %eax
	jmp	.L341
.L333:
	movl	$0, %eax
	jmp	.L341
.L338:
	cmpq	$0, -56(%rbp)
	jne	.L342
	movq	$25, -8(%rbp)
	jmp	.L340
.L342:
	movq	$20, -8(%rbp)
	jmp	.L340
.L323:
	movl	$0, %eax
	jmp	.L341
.L337:
	movl	-44(%rbp), %eax
	imull	-60(%rbp), %eax
	movl	%eax, %edx
	movl	-32(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -44(%rbp)
	addl	$1, -36(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L340
.L329:
	movl	$0, -44(%rbp)
	movl	$1, -40(%rbp)
	movl	$0, -36(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L340
.L322:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-45(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L344
	movq	$17, -8(%rbp)
	jmp	.L340
.L344:
	movq	$11, -8(%rbp)
	jmp	.L340
.L325:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L346
	movq	$13, -8(%rbp)
	jmp	.L340
.L346:
	movq	$26, -8(%rbp)
	jmp	.L340
.L320:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L348
	movq	$18, -8(%rbp)
	jmp	.L340
.L348:
	movq	$22, -8(%rbp)
	jmp	.L340
.L331:
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L340
.L332:
	movl	$0, %eax
	jmp	.L341
.L330:
	movl	$-1, -40(%rbp)
	addl	$1, -36(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L340
.L328:
	movsbl	-45(%rbp), %eax
	subl	$48, %eax
	movl	%eax, -32(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L340
.L335:
	cmpl	$36, -60(%rbp)
	jle	.L350
	movq	$23, -8(%rbp)
	jmp	.L340
.L350:
	movq	$16, -8(%rbp)
	jmp	.L340
.L318:
	movl	$0, %eax
	jmp	.L341
.L324:
	movl	-44(%rbp), %eax
	imull	-40(%rbp), %eax
	jmp	.L341
.L336:
	movsbl	-45(%rbp), %eax
	movl	%eax, %edi
	call	toupper@PLT
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	subl	$55, %eax
	movl	%eax, -32(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L340
.L339:
	movl	-32(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jl	.L352
	movq	$9, -8(%rbp)
	jmp	.L340
.L352:
	movq	$3, -8(%rbp)
	jmp	.L340
.L334:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-45(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L354
	movq	$5, -8(%rbp)
	jmp	.L340
.L354:
	movq	$8, -8(%rbp)
	jmp	.L340
.L326:
	cmpl	$1, -60(%rbp)
	jg	.L356
	movq	$27, -8(%rbp)
	jmp	.L340
.L356:
	movq	$6, -8(%rbp)
	jmp	.L340
.L359:
	nop
.L340:
	jmp	.L358
.L341:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	str_to_int, .-str_to_int
	.section	.rodata
.LC15:
	.string	"Roman: %Ro\n"
.LC16:
	.string	"Zeckendorf: %Zr\n"
.LC17:
	.string	"Base 16: %Cv\n"
.LC18:
	.string	"Base 16 (uppercase): %CV\n"
.LC19:
	.string	"ff"
.LC20:
	.string	"String to int: %to\n"
.LC21:
	.string	"FF"
	.align 8
.LC22:
	.string	"String to int (uppercase): %TO\n"
.LC23:
	.string	"Memory dump (int): %mi\n"
	.align 8
.LC24:
	.string	"Memory dump (unsigned int): %mu\n"
.LC26:
	.string	"Memory dump (double): %md\n"
.LC28:
	.string	"Memory dump (float): %mf\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB12:
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
	movq	$0, _TIG_IZ_VUiJ_envp(%rip)
	nop
.L361:
	movq	$0, _TIG_IZ_VUiJ_argv(%rip)
	nop
.L362:
	movl	$0, _TIG_IZ_VUiJ_argc(%rip)
	nop
	nop
.L363:
.L364:
#APP
# 111 "Kirsaxg1_02-Programming_9.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-VUiJ--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_VUiJ_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_VUiJ_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_VUiJ_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L370:
	cmpq	$2, -8(%rbp)
	je	.L365
	cmpq	$2, -8(%rbp)
	ja	.L372
	cmpq	$0, -8(%rbp)
	je	.L367
	cmpq	$1, -8(%rbp)
	jne	.L372
	movq	$0, -8(%rbp)
	jmp	.L368
.L367:
	movl	$2023, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	overprintf
	movl	$123, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	overprintf
	movl	$16, %edx
	movl	$255, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	overprintf
	movl	$16, %edx
	movl	$255, %esi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	overprintf
	movl	$16, %edx
	leaq	.LC19(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	overprintf
	movl	$16, %edx
	leaq	.LC21(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	overprintf
	movl	$123, %esi
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	overprintf
	movl	$123, %esi
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	overprintf
	movq	.LC25(%rip), %rax
	movq	%rax, %xmm0
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	overprintf
	movq	.LC27(%rip), %rax
	movq	%rax, %xmm0
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	overprintf
	movq	$2, -8(%rbp)
	jmp	.L368
.L365:
	movl	$0, %eax
	jmp	.L371
.L372:
	nop
.L368:
	jmp	.L370
.L371:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC25:
	.long	446676599
	.long	1079958831
	.align 8
.LC27:
	.long	536870912
	.long	1079958831
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
