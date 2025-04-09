	.file	"JoezerSmaniotto_L-gica-de-Programa-o-C_8_flatten.c"
	.text
	.globl	_TIG_IZ_PRNu_envp
	.bss
	.align 8
	.type	_TIG_IZ_PRNu_envp, @object
	.size	_TIG_IZ_PRNu_envp, 8
_TIG_IZ_PRNu_envp:
	.zero	8
	.globl	_TIG_IZ_PRNu_argc
	.align 4
	.type	_TIG_IZ_PRNu_argc, @object
	.size	_TIG_IZ_PRNu_argc, 4
_TIG_IZ_PRNu_argc:
	.zero	4
	.globl	_TIG_IZ_PRNu_argv
	.align 8
	.type	_TIG_IZ_PRNu_argv, @object
	.size	_TIG_IZ_PRNu_argv, 8
_TIG_IZ_PRNu_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Informe o Valor a Ser Pesquisado: "
.LC1:
	.string	"%d"
	.align 8
.LC2:
	.string	"Informe o Valor na Posi\303\247\303\243o %d: "
.LC3:
	.string	"Codigo Nao Existe"
.LC4:
	.string	"%d\n"
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_PRNu_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_PRNu_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_PRNu_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 128 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-PRNu--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_PRNu_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_PRNu_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_PRNu_envp(%rip)
	nop
	movq	$7, -56(%rbp)
.L35:
	cmpq	$24, -56(%rbp)
	ja	.L38
	movq	-56(%rbp), %rax
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
	.long	.L22-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L38-.L8
	.long	.L16-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L38-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L38-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	movl	$0, -60(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-68(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$19, -56(%rbp)
	jmp	.L23
.L15:
	cmpl	$7, -64(%rbp)
	jg	.L24
	movq	$16, -56(%rbp)
	jmp	.L23
.L24:
	movq	$22, -56(%rbp)
	jmp	.L23
.L18:
	movl	-64(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rdx
	movl	-64(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -64(%rbp)
	movq	$9, -56(%rbp)
	jmp	.L23
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$18, -56(%rbp)
	jmp	.L23
.L14:
	movl	-64(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %edx
	movl	-68(%rbp), %eax
	cmpl	%eax, %edx
	jne	.L26
	movq	$11, -56(%rbp)
	jmp	.L23
.L26:
	movq	$21, -56(%rbp)
	jmp	.L23
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-68(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$19, -56(%rbp)
	jmp	.L23
.L11:
	addl	$1, -64(%rbp)
	movq	$15, -56(%rbp)
	jmp	.L23
.L16:
	movl	-64(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, -60(%rbp)
	movq	$21, -56(%rbp)
	jmp	.L23
.L17:
	cmpl	$7, -64(%rbp)
	jg	.L28
	movq	$8, -56(%rbp)
	jmp	.L23
.L28:
	movq	$24, -56(%rbp)
	jmp	.L23
.L12:
	movl	-68(%rbp), %eax
	testl	%eax, %eax
	jle	.L30
	movq	$6, -56(%rbp)
	jmp	.L23
.L30:
	movq	$5, -56(%rbp)
	jmp	.L23
.L20:
	movl	$0, -64(%rbp)
	movq	$15, -56(%rbp)
	jmp	.L23
.L10:
	cmpl	$0, -60(%rbp)
	jne	.L32
	movq	$23, -56(%rbp)
	jmp	.L23
.L32:
	movq	$18, -56(%rbp)
	jmp	.L23
.L21:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L36
	jmp	.L37
.L22:
	movl	$0, -60(%rbp)
	movl	$0, -64(%rbp)
	movq	$9, -56(%rbp)
	jmp	.L23
.L19:
	movq	$0, -56(%rbp)
	jmp	.L23
.L38:
	nop
.L23:
	jmp	.L35
.L37:
	call	__stack_chk_fail@PLT
.L36:
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
