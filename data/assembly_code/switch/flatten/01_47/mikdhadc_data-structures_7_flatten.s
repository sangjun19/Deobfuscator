	.file	"mikdhadc_data-structures_7_flatten.c"
	.text
	.globl	_TIG_IZ_SVzM_argv
	.bss
	.align 8
	.type	_TIG_IZ_SVzM_argv, @object
	.size	_TIG_IZ_SVzM_argv, 8
_TIG_IZ_SVzM_argv:
	.zero	8
	.globl	_TIG_IZ_SVzM_argc
	.align 4
	.type	_TIG_IZ_SVzM_argc, @object
	.size	_TIG_IZ_SVzM_argc, 4
_TIG_IZ_SVzM_argc:
	.zero	4
	.globl	_TIG_IZ_SVzM_envp
	.align 8
	.type	_TIG_IZ_SVzM_envp, @object
	.size	_TIG_IZ_SVzM_envp, 8
_TIG_IZ_SVzM_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Do you want to continue : "
.LC1:
	.string	" %c"
.LC2:
	.string	"Enter : "
.LC3:
	.string	"%d"
.LC4:
	.string	"Deleted."
.LC5:
	.string	"\nkey not found"
.LC6:
	.string	"\nEnter key : "
.LC7:
	.string	"Enter no.of elements : "
.LC8:
	.string	"\nDELETION :"
	.align 8
.LC9:
	.string	"\n1.From beginning \n2.From end \n3.After any key"
.LC10:
	.string	"%d\n"
.LC11:
	.string	"\n New list :"
.LC12:
	.string	"\nEnter your choice(1/2/3) : "
.LC13:
	.string	"\nEmpty list."
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_SVzM_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_SVzM_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_SVzM_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 136 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-SVzM--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_SVzM_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_SVzM_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_SVzM_envp(%rip)
	nop
	movq	$55, -40(%rbp)
.L64:
	cmpq	$59, -40(%rbp)
	ja	.L67
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L38-.L8
	.long	.L67-.L8
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L67-.L8
	.long	.L28-.L8
	.long	.L67-.L8
	.long	.L27-.L8
	.long	.L67-.L8
	.long	.L26-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L23-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L18-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L67-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L67-.L8
	.long	.L12-.L8
	.long	.L68-.L8
	.long	.L67-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L67-.L8
	.long	.L7-.L8
	.text
.L13:
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L39
	movq	$38, -40(%rbp)
	jmp	.L41
.L39:
	movq	$54, -40(%rbp)
	jmp	.L41
.L37:
	cmpq	$0, -64(%rbp)
	jne	.L43
	movq	$36, -40(%rbp)
	jmp	.L41
.L43:
	movq	$28, -40(%rbp)
	jmp	.L41
.L32:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-85(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$35, -40(%rbp)
	jmp	.L41
.L16:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-80(%rbp), %edx
	movq	-48(%rbp), %rax
	movl	%edx, (%rax)
	movq	-48(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-64(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	addl	$1, -68(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L41
.L10:
	movq	-48(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -40(%rbp)
	jmp	.L41
.L26:
	movl	-84(%rbp), %eax
	cmpl	%eax, -68(%rbp)
	jge	.L45
	movq	$45, -40(%rbp)
	jmp	.L41
.L45:
	movq	$13, -40(%rbp)
	jmp	.L41
.L30:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L47
	movq	$2, -40(%rbp)
	jmp	.L41
.L47:
	movq	$41, -40(%rbp)
	jmp	.L41
.L27:
	movq	$12, -40(%rbp)
	jmp	.L41
.L21:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -40(%rbp)
	jmp	.L41
.L33:
	movl	-76(%rbp), %eax
	cmpl	$3, %eax
	je	.L49
	cmpl	$3, %eax
	jg	.L50
	cmpl	$1, %eax
	je	.L51
	cmpl	$2, %eax
	je	.L52
	jmp	.L50
.L49:
	movq	$32, -40(%rbp)
	jmp	.L53
.L52:
	movq	$9, -40(%rbp)
	jmp	.L53
.L51:
	movq	$29, -40(%rbp)
	jmp	.L53
.L50:
	movq	$21, -40(%rbp)
	nop
.L53:
	jmp	.L41
.L35:
	movq	-56(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	$49, -40(%rbp)
	jmp	.L41
.L31:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	$51, -40(%rbp)
	jmp	.L41
.L12:
	cmpq	$0, -64(%rbp)
	je	.L54
	movq	$10, -40(%rbp)
	jmp	.L41
.L54:
	movq	$59, -40(%rbp)
	jmp	.L41
.L28:
	cmpq	$0, -64(%rbp)
	je	.L56
	movq	$44, -40(%rbp)
	jmp	.L41
.L56:
	movq	$52, -40(%rbp)
	jmp	.L41
.L23:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L41
.L29:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movl	$0, (%rax)
	movq	-56(%rbp), %rax
	movq	$0, 8(%rax)
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-84(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -68(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L41
.L9:
	movq	$17, -40(%rbp)
	jmp	.L41
.L7:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$37, -40(%rbp)
	jmp	.L41
.L19:
	movq	-64(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	$49, -40(%rbp)
	jmp	.L41
.L14:
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	$41, -40(%rbp)
	jmp	.L41
.L25:
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -40(%rbp)
	jmp	.L41
.L17:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	$19, -40(%rbp)
	jmp	.L41
.L36:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -40(%rbp)
	jmp	.L41
.L20:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-76(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$11, -40(%rbp)
	jmp	.L41
.L18:
	movq	-64(%rbp), %rax
	movl	(%rax), %edx
	movl	-72(%rbp), %eax
	cmpl	%eax, %edx
	je	.L58
	movq	$46, -40(%rbp)
	jmp	.L41
.L58:
	movq	$4, -40(%rbp)
	jmp	.L41
.L34:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	$51, -40(%rbp)
	jmp	.L41
.L15:
	cmpq	$0, -64(%rbp)
	je	.L60
	movq	$48, -40(%rbp)
	jmp	.L41
.L60:
	movq	$4, -40(%rbp)
	jmp	.L41
.L22:
	movzbl	-85(%rbp), %eax
	cmpb	$121, %al
	jne	.L62
	movq	$37, -40(%rbp)
	jmp	.L41
.L62:
	movq	$5, -40(%rbp)
	jmp	.L41
.L24:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -40(%rbp)
	jmp	.L41
.L38:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -40(%rbp)
	jmp	.L41
.L67:
	nop
.L41:
	jmp	.L64
.L68:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L66
	call	__stack_chk_fail@PLT
.L66:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
