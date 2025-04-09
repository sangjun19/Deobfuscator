	.file	"aniruddh-joshi_OS_Lab_Z_flatten.c"
	.text
	.globl	_TIG_IZ_h4xb_argv
	.bss
	.align 8
	.type	_TIG_IZ_h4xb_argv, @object
	.size	_TIG_IZ_h4xb_argv, 8
_TIG_IZ_h4xb_argv:
	.zero	8
	.globl	_TIG_IZ_h4xb_argc
	.align 4
	.type	_TIG_IZ_h4xb_argc, @object
	.size	_TIG_IZ_h4xb_argc, 4
_TIG_IZ_h4xb_argc:
	.zero	4
	.globl	_TIG_IZ_h4xb_envp
	.align 8
	.type	_TIG_IZ_h4xb_envp, @object
	.size	_TIG_IZ_h4xb_envp, 8
_TIG_IZ_h4xb_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"ls"
	.text
	.globl	my_ls
	.type	my_ls, @function
my_ls:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L7
	cmpq	$1, -8(%rbp)
	jne	.L8
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	system@PLT
	movq	$0, -8(%rbp)
	jmp	.L4
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	my_ls, .-my_ls
	.section	.rodata
.LC1:
	.string	"Unsupported command: %s\n"
.LC2:
	.string	"cp"
.LC3:
	.string	"Usage: %s command [options]\n"
	.align 8
.LC4:
	.string	"Usage: %s grep pattern filename\n"
	.align 8
.LC5:
	.string	"Usage: %s cp source destination\n"
.LC6:
	.string	"grep"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_h4xb_envp(%rip)
	nop
.L10:
	movq	$0, _TIG_IZ_h4xb_argv(%rip)
	nop
.L11:
	movl	$0, _TIG_IZ_h4xb_argc(%rip)
	nop
	nop
.L12:
.L13:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-h4xb--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_h4xb_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_h4xb_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_h4xb_envp(%rip)
	nop
	movq	$4, -8(%rbp)
.L51:
	cmpq	$20, -8(%rbp)
	ja	.L52
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L16(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L16(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L16:
	.long	.L36-.L16
	.long	.L35-.L16
	.long	.L34-.L16
	.long	.L33-.L16
	.long	.L32-.L16
	.long	.L31-.L16
	.long	.L30-.L16
	.long	.L29-.L16
	.long	.L28-.L16
	.long	.L27-.L16
	.long	.L26-.L16
	.long	.L25-.L16
	.long	.L24-.L16
	.long	.L23-.L16
	.long	.L22-.L16
	.long	.L21-.L16
	.long	.L20-.L16
	.long	.L19-.L16
	.long	.L18-.L16
	.long	.L17-.L16
	.long	.L15-.L16
	.text
.L18:
	movq	-48(%rbp), %rax
	addq	$24, %rax
	movq	(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	my_grep
	movq	$1, -8(%rbp)
	jmp	.L37
.L32:
	cmpl	$1, -36(%rbp)
	jg	.L38
	movq	$9, -8(%rbp)
	jmp	.L37
.L38:
	movq	$16, -8(%rbp)
	jmp	.L37
.L22:
	cmpl	$0, -20(%rbp)
	jne	.L40
	movq	$19, -8(%rbp)
	jmp	.L37
.L40:
	movq	$12, -8(%rbp)
	jmp	.L37
.L21:
	movl	$1, %eax
	jmp	.L42
.L24:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$20, -8(%rbp)
	jmp	.L37
.L28:
	cmpl	$4, -36(%rbp)
	je	.L43
	movq	$0, -8(%rbp)
	jmp	.L37
.L43:
	movq	$3, -8(%rbp)
	jmp	.L37
.L35:
	movl	$0, %eax
	jmp	.L42
.L33:
	movq	-48(%rbp), %rax
	addq	$24, %rax
	movq	(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	my_cp
	movq	$1, -8(%rbp)
	jmp	.L37
.L20:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L37
.L25:
	call	my_ls
	movq	$1, -8(%rbp)
	jmp	.L37
.L27:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -8(%rbp)
	jmp	.L37
.L23:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L37
.L17:
	cmpl	$4, -36(%rbp)
	je	.L45
	movq	$17, -8(%rbp)
	jmp	.L37
.L45:
	movq	$18, -8(%rbp)
	jmp	.L37
.L19:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L37
.L30:
	movl	$1, %eax
	jmp	.L42
.L31:
	movl	$1, %eax
	jmp	.L42
.L26:
	cmpl	$0, -16(%rbp)
	jne	.L47
	movq	$11, -8(%rbp)
	jmp	.L37
.L47:
	movq	$7, -8(%rbp)
	jmp	.L37
.L36:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -8(%rbp)
	jmp	.L37
.L29:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L37
.L34:
	cmpl	$0, -12(%rbp)
	jne	.L49
	movq	$8, -8(%rbp)
	jmp	.L37
.L49:
	movq	$13, -8(%rbp)
	jmp	.L37
.L15:
	movl	$1, %eax
	jmp	.L42
.L52:
	nop
.L37:
	jmp	.L51
.L42:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
.LC7:
	.string	"r"
.LC8:
	.string	"w"
	.align 8
.LC9:
	.string	"Error opening destination file"
.LC10:
	.string	"File copied successfully!"
.LC11:
	.string	"Error opening source file"
	.text
	.globl	my_cp
	.type	my_cp, @function
my_cp:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$14, -8(%rbp)
.L77:
	cmpq	$17, -8(%rbp)
	ja	.L78
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L56(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L56(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L56:
	.long	.L79-.L56
	.long	.L67-.L56
	.long	.L66-.L56
	.long	.L65-.L56
	.long	.L64-.L56
	.long	.L79-.L56
	.long	.L62-.L56
	.long	.L78-.L56
	.long	.L61-.L56
	.long	.L78-.L56
	.long	.L78-.L56
	.long	.L60-.L56
	.long	.L78-.L56
	.long	.L59-.L56
	.long	.L58-.L56
	.long	.L78-.L56
	.long	.L57-.L56
	.long	.L79-.L56
	.text
.L64:
	cmpq	$0, -16(%rbp)
	jne	.L69
	movq	$3, -8(%rbp)
	jmp	.L71
.L69:
	movq	$16, -8(%rbp)
	jmp	.L71
.L58:
	movq	-40(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -24(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L71
.L61:
	movsbl	-29(%rbp), %eax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	fputc@PLT
	movq	$16, -8(%rbp)
	jmp	.L71
.L67:
	movq	-48(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L71
.L65:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$5, -8(%rbp)
	jmp	.L71
.L57:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movb	%al, -29(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L71
.L60:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$17, -8(%rbp)
	jmp	.L71
.L59:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$0, -8(%rbp)
	jmp	.L71
.L62:
	cmpq	$0, -24(%rbp)
	jne	.L73
	movq	$13, -8(%rbp)
	jmp	.L71
.L73:
	movq	$1, -8(%rbp)
	jmp	.L71
.L66:
	cmpb	$-1, -29(%rbp)
	je	.L75
	movq	$8, -8(%rbp)
	jmp	.L71
.L75:
	movq	$11, -8(%rbp)
	jmp	.L71
.L78:
	nop
.L71:
	jmp	.L77
.L79:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	my_cp, .-my_cp
	.section	.rodata
.LC12:
	.string	"Error opening file"
.LC13:
	.string	"%s"
	.text
	.globl	my_grep
	.type	my_grep, @function
my_grep:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%rdi, -152(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$7, -120(%rbp)
.L102:
	cmpq	$12, -120(%rbp)
	ja	.L105
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L83(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L83(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L83:
	.long	.L93-.L83
	.long	.L92-.L83
	.long	.L105-.L83
	.long	.L91-.L83
	.long	.L105-.L83
	.long	.L90-.L83
	.long	.L89-.L83
	.long	.L88-.L83
	.long	.L87-.L83
	.long	.L106-.L83
	.long	.L106-.L83
	.long	.L84-.L83
	.long	.L82-.L83
	.text
.L82:
	cmpq	$0, -128(%rbp)
	je	.L94
	movq	$1, -120(%rbp)
	jmp	.L96
.L94:
	movq	$3, -120(%rbp)
	jmp	.L96
.L87:
	cmpq	$0, -136(%rbp)
	je	.L97
	movq	$6, -120(%rbp)
	jmp	.L96
.L97:
	movq	$0, -120(%rbp)
	jmp	.L96
.L92:
	movq	-152(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -136(%rbp)
	movq	$8, -120(%rbp)
	jmp	.L96
.L91:
	movq	-144(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$9, -120(%rbp)
	jmp	.L96
.L84:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$10, -120(%rbp)
	jmp	.L96
.L89:
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -120(%rbp)
	jmp	.L96
.L90:
	cmpq	$0, -144(%rbp)
	jne	.L100
	movq	$11, -120(%rbp)
	jmp	.L96
.L100:
	movq	$0, -120(%rbp)
	jmp	.L96
.L93:
	movq	-144(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movl	$100, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -128(%rbp)
	movq	$12, -120(%rbp)
	jmp	.L96
.L88:
	movq	-160(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -144(%rbp)
	movq	$5, -120(%rbp)
	jmp	.L96
.L105:
	nop
.L96:
	jmp	.L102
.L106:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L104
	call	__stack_chk_fail@PLT
.L104:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	my_grep, .-my_grep
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
