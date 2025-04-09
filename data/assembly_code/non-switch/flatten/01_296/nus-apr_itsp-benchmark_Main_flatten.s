	.file	"nus-apr_itsp-benchmark_Main_flatten.c"
	.text
	.globl	_TIG_IZ_Rdn1_envp
	.bss
	.align 8
	.type	_TIG_IZ_Rdn1_envp, @object
	.size	_TIG_IZ_Rdn1_envp, 8
_TIG_IZ_Rdn1_envp:
	.zero	8
	.globl	_TIG_IZ_Rdn1_argv
	.align 8
	.type	_TIG_IZ_Rdn1_argv, @object
	.size	_TIG_IZ_Rdn1_argv, 8
_TIG_IZ_Rdn1_argv:
	.zero	8
	.globl	_TIG_IZ_Rdn1_argc
	.align 4
	.type	_TIG_IZ_Rdn1_argc, @object
	.size	_TIG_IZ_Rdn1_argc, 4
_TIG_IZ_Rdn1_argc:
	.zero	4
	.text
	.globl	reverse
	.type	reverse, @function
reverse:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	$7, -16(%rbp)
.L14:
	cmpq	$7, -16(%rbp)
	ja	.L15
	movq	-16(%rbp), %rax
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
	.long	.L9-.L4
	.long	.L15-.L4
	.long	.L8-.L4
	.long	.L15-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L16-.L4
	.long	.L3-.L4
	.text
.L7:
	cmpq	$0, -24(%rbp)
	je	.L10
	movq	$0, -16(%rbp)
	jmp	.L12
.L10:
	movq	$2, -16(%rbp)
	jmp	.L12
.L6:
	movq	$0, -32(%rbp)
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L12
.L9:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L12
.L3:
	movq	$5, -16(%rbp)
	jmp	.L12
.L8:
	movq	-40(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$6, -16(%rbp)
	jmp	.L12
.L15:
	nop
.L12:
	jmp	.L14
.L16:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	reverse, .-reverse
	.globl	push
	.type	push, @function
push:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, %eax
	movb	%al, -44(%rbp)
	movq	$2, -24(%rbp)
.L23:
	cmpq	$2, -24(%rbp)
	je	.L18
	cmpq	$2, -24(%rbp)
	ja	.L25
	cmpq	$0, -24(%rbp)
	je	.L20
	cmpq	$1, -24(%rbp)
	jne	.L25
	jmp	.L24
.L20:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movzbl	-44(%rbp), %edx
	movb	%dl, (%rax)
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$1, -24(%rbp)
	jmp	.L22
.L18:
	movq	$0, -24(%rbp)
	jmp	.L22
.L25:
	nop
.L22:
	jmp	.L23
.L24:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	push, .-push
	.section	.rodata
.LC0:
	.string	"NULL"
.LC1:
	.string	"%c->"
	.text
	.globl	printList
	.type	printList, @function
printList:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$4, -8(%rbp)
.L38:
	cmpq	$6, -8(%rbp)
	ja	.L39
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L29(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L29(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L29:
	.long	.L39-.L29
	.long	.L33-.L29
	.long	.L40-.L29
	.long	.L31-.L29
	.long	.L30-.L29
	.long	.L39-.L29
	.long	.L28-.L29
	.text
.L30:
	movq	$3, -8(%rbp)
	jmp	.L34
.L33:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L34
.L31:
	cmpq	$0, -24(%rbp)
	je	.L35
	movq	$6, -8(%rbp)
	jmp	.L34
.L35:
	movq	$1, -8(%rbp)
	jmp	.L34
.L28:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L34
.L39:
	nop
.L34:
	jmp	.L38
.L40:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	printList, .-printList
	.globl	isPalindrome
	.type	isPalindrome, @function
isPalindrome:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -16(%rbp)
.L72:
	cmpq	$20, -16(%rbp)
	ja	.L75
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L44(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L44(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L44:
	.long	.L57-.L44
	.long	.L56-.L44
	.long	.L55-.L44
	.long	.L54-.L44
	.long	.L53-.L44
	.long	.L75-.L44
	.long	.L52-.L44
	.long	.L75-.L44
	.long	.L51-.L44
	.long	.L75-.L44
	.long	.L75-.L44
	.long	.L50-.L44
	.long	.L75-.L44
	.long	.L49-.L44
	.long	.L75-.L44
	.long	.L48-.L44
	.long	.L47-.L44
	.long	.L46-.L44
	.long	.L45-.L44
	.long	.L75-.L44
	.long	.L43-.L44
	.text
.L45:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L58
	movq	$3, -16(%rbp)
	jmp	.L60
.L58:
	movq	$1, -16(%rbp)
	jmp	.L60
.L53:
	movq	$17, -16(%rbp)
	jmp	.L60
.L48:
	cmpq	$0, -40(%rbp)
	je	.L61
	movq	$18, -16(%rbp)
	jmp	.L60
.L61:
	movq	$1, -16(%rbp)
	jmp	.L60
.L51:
	movq	-48(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	-32(%rbp), %rax
	movq	$0, 8(%rax)
	leaq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	reverse
	movq	-56(%rbp), %rdx
	movq	-72(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	compareLists
	movl	%eax, -60(%rbp)
	leaq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	reverse
	movq	$20, -16(%rbp)
	jmp	.L60
.L56:
	cmpq	$0, -40(%rbp)
	je	.L63
	movq	$13, -16(%rbp)
	jmp	.L60
.L63:
	movq	$8, -16(%rbp)
	jmp	.L60
.L54:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -48(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L60
.L47:
	movq	-32(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-56(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	$11, -16(%rbp)
	jmp	.L60
.L50:
	movl	-60(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L73
	jmp	.L74
.L49:
	movq	-48(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -48(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L60
.L46:
	movq	-72(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$0, -24(%rbp)
	movl	$1, -60(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L60
.L52:
	cmpq	$0, -72(%rbp)
	je	.L66
	movq	$0, -16(%rbp)
	jmp	.L60
.L66:
	movq	$11, -16(%rbp)
	jmp	.L60
.L57:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L68
	movq	$15, -16(%rbp)
	jmp	.L60
.L68:
	movq	$11, -16(%rbp)
	jmp	.L60
.L55:
	movq	-56(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	$11, -16(%rbp)
	jmp	.L60
.L43:
	cmpq	$0, -24(%rbp)
	je	.L70
	movq	$16, -16(%rbp)
	jmp	.L60
.L70:
	movq	$2, -16(%rbp)
	jmp	.L60
.L75:
	nop
.L60:
	jmp	.L72
.L74:
	call	__stack_chk_fail@PLT
.L73:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	isPalindrome, .-isPalindrome
	.section	.rodata
.LC2:
	.string	"Yes"
.LC3:
	.string	"No"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Rdn1_envp(%rip)
	nop
.L77:
	movq	$0, _TIG_IZ_Rdn1_argv(%rip)
	nop
.L78:
	movl	$0, _TIG_IZ_Rdn1_argc(%rip)
	nop
	nop
.L79:
.L80:
#APP
# 160 "nus-apr_itsp-benchmark_Main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Rdn1--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_Rdn1_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_Rdn1_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_Rdn1_envp(%rip)
	nop
	movq	$2, -16(%rbp)
.L98:
	cmpq	$11, -16(%rbp)
	ja	.L101
	movq	-16(%rbp), %rax
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
	.long	.L101-.L83
	.long	.L91-.L83
	.long	.L90-.L83
	.long	.L101-.L83
	.long	.L89-.L83
	.long	.L88-.L83
	.long	.L87-.L83
	.long	.L86-.L83
	.long	.L85-.L83
	.long	.L101-.L83
	.long	.L84-.L83
	.long	.L82-.L83
	.text
.L89:
	movq	$0, -24(%rbp)
	call	getchar@PLT
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, -36(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L92
.L85:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$11, -16(%rbp)
	jmp	.L92
.L91:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$11, -16(%rbp)
	jmp	.L92
.L82:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L99
	jmp	.L100
.L87:
	movl	-36(%rbp), %eax
	movsbl	%al, %edx
	leaq	-24(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	push
	call	getchar@PLT
	movl	%eax, -36(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L92
.L88:
	cmpl	$0, -32(%rbp)
	je	.L94
	movq	$8, -16(%rbp)
	jmp	.L92
.L94:
	movq	$1, -16(%rbp)
	jmp	.L92
.L84:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	isPalindrome
	movl	%eax, -32(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L92
.L86:
	cmpl	$10, -36(%rbp)
	je	.L96
	movq	$6, -16(%rbp)
	jmp	.L92
.L96:
	movq	$10, -16(%rbp)
	jmp	.L92
.L90:
	movq	$4, -16(%rbp)
	jmp	.L92
.L101:
	nop
.L92:
	jmp	.L98
.L100:
	call	__stack_chk_fail@PLT
.L99:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	compareLists
	.type	compareLists, @function
compareLists:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$9, -8(%rbp)
.L128:
	cmpq	$13, -8(%rbp)
	ja	.L129
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L105(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L105(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L105:
	.long	.L115-.L105
	.long	.L114-.L105
	.long	.L113-.L105
	.long	.L129-.L105
	.long	.L112-.L105
	.long	.L111-.L105
	.long	.L110-.L105
	.long	.L109-.L105
	.long	.L129-.L105
	.long	.L108-.L105
	.long	.L129-.L105
	.long	.L107-.L105
	.long	.L106-.L105
	.long	.L104-.L105
	.text
.L112:
	cmpq	$0, -16(%rbp)
	je	.L116
	movq	$13, -8(%rbp)
	jmp	.L118
.L116:
	movq	$0, -8(%rbp)
	jmp	.L118
.L106:
	movl	$0, %eax
	jmp	.L119
.L114:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L118
.L107:
	cmpq	$0, -24(%rbp)
	je	.L120
	movq	$4, -8(%rbp)
	jmp	.L118
.L120:
	movq	$0, -8(%rbp)
	jmp	.L118
.L108:
	movq	$6, -8(%rbp)
	jmp	.L118
.L104:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L122
	movq	$1, -8(%rbp)
	jmp	.L118
.L122:
	movq	$7, -8(%rbp)
	jmp	.L118
.L110:
	movq	-40(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L118
.L111:
	cmpq	$0, -16(%rbp)
	jne	.L124
	movq	$2, -8(%rbp)
	jmp	.L118
.L124:
	movq	$12, -8(%rbp)
	jmp	.L118
.L115:
	cmpq	$0, -24(%rbp)
	jne	.L126
	movq	$5, -8(%rbp)
	jmp	.L118
.L126:
	movq	$12, -8(%rbp)
	jmp	.L118
.L109:
	movl	$0, %eax
	jmp	.L119
.L113:
	movl	$1, %eax
	jmp	.L119
.L129:
	nop
.L118:
	jmp	.L128
.L119:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	compareLists, .-compareLists
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
