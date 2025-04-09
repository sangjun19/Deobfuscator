	.file	"prabhanjanbhat_DS-lab_1_flatten.c"
	.text
	.globl	_TIG_IZ_0xkN_envp
	.bss
	.align 8
	.type	_TIG_IZ_0xkN_envp, @object
	.size	_TIG_IZ_0xkN_envp, 8
_TIG_IZ_0xkN_envp:
	.zero	8
	.globl	_TIG_IZ_0xkN_argv
	.align 8
	.type	_TIG_IZ_0xkN_argv, @object
	.size	_TIG_IZ_0xkN_argv, 8
_TIG_IZ_0xkN_argv:
	.zero	8
	.globl	_TIG_IZ_0xkN_argc
	.align 4
	.type	_TIG_IZ_0xkN_argc, @object
	.size	_TIG_IZ_0xkN_argc, 4
_TIG_IZ_0xkN_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Withdrawal successful. New balance: %.2f\n"
	.align 8
.LC1:
	.string	"Insufficient funds. Withdrawal failed."
	.text
	.globl	withdraw
	.type	withdraw, @function
withdraw:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movss	%xmm0, -28(%rbp)
	movq	$0, -8(%rbp)
.L12:
	cmpq	$4, -8(%rbp)
	je	.L2
	cmpq	$4, -8(%rbp)
	ja	.L15
	cmpq	$3, -8(%rbp)
	je	.L16
	cmpq	$3, -8(%rbp)
	ja	.L15
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$2, -8(%rbp)
	je	.L6
	jmp	.L15
.L2:
	movq	-24(%rbp), %rax
	movss	4(%rax), %xmm0
	subss	-28(%rbp), %xmm0
	movq	-24(%rbp), %rax
	movss	%xmm0, 4(%rax)
	movq	-24(%rbp), %rax
	movss	4(%rax), %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	%xmm0, %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L7
.L5:
	movq	-24(%rbp), %rax
	movss	4(%rax), %xmm1
	movss	-28(%rbp), %xmm0
	comiss	%xmm1, %xmm0
	jbe	.L14
	movq	$2, -8(%rbp)
	jmp	.L7
.L14:
	movq	$4, -8(%rbp)
	jmp	.L7
.L6:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L7
.L15:
	nop
.L7:
	jmp	.L12
.L16:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	withdraw, .-withdraw
	.section	.rodata
	.align 8
.LC3:
	.string	"Account created successfully. Account Number: %d\n"
	.text
	.globl	createAccount
	.type	createAccount, @function
createAccount:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$2, -8(%rbp)
.L23:
	cmpq	$2, -8(%rbp)
	je	.L18
	cmpq	$2, -8(%rbp)
	ja	.L25
	cmpq	$0, -8(%rbp)
	je	.L20
	cmpq	$1, -8(%rbp)
	jne	.L25
	jmp	.L24
.L20:
	movq	-24(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	pxor	%xmm0, %xmm0
	movss	%xmm0, 4(%rax)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L22
.L18:
	movq	$0, -8(%rbp)
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
.LFE6:
	.size	createAccount, .-createAccount
	.section	.rodata
.LC4:
	.string	"Account Balance: %.2f\n"
	.text
	.globl	balanceInquiry
	.type	balanceInquiry, @function
balanceInquiry:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L31:
	cmpq	$0, -8(%rbp)
	je	.L32
	cmpq	$1, -8(%rbp)
	jne	.L33
	movq	-24(%rbp), %rax
	movss	4(%rax), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L29
.L33:
	nop
.L29:
	jmp	.L31
.L32:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	balanceInquiry, .-balanceInquiry
	.section	.rodata
	.align 8
.LC5:
	.string	"Deposit successful. New balance: %.2f\n"
	.text
	.globl	deposit
	.type	deposit, @function
deposit:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movss	%xmm0, -28(%rbp)
	movq	$2, -8(%rbp)
.L40:
	cmpq	$2, -8(%rbp)
	je	.L35
	cmpq	$2, -8(%rbp)
	ja	.L41
	cmpq	$0, -8(%rbp)
	je	.L42
	cmpq	$1, -8(%rbp)
	jne	.L41
	movq	-24(%rbp), %rax
	movss	4(%rax), %xmm0
	addss	-28(%rbp), %xmm0
	movq	-24(%rbp), %rax
	movss	%xmm0, 4(%rax)
	movq	-24(%rbp), %rax
	movss	4(%rax), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L38
.L35:
	movq	$1, -8(%rbp)
	jmp	.L38
.L41:
	nop
.L38:
	jmp	.L40
.L42:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	deposit, .-deposit
	.section	.rodata
.LC6:
	.string	"Enter withdrawal amount: "
.LC7:
	.string	"%f"
.LC8:
	.string	"\n1. Deposit"
.LC9:
	.string	"2. Withdraw"
.LC10:
	.string	"3. Balance Inquiry"
.LC11:
	.string	"4. Exit"
.LC12:
	.string	"Enter your choice: "
.LC13:
	.string	"%d"
.LC14:
	.string	"Enter deposit amount: "
.LC15:
	.string	"Exiting the program. Goodbye!"
	.align 8
.LC16:
	.string	"Invalid choice. Please enter a valid option."
.LC17:
	.string	"Enter Account Number: "
	.text
	.globl	main
	.type	main, @function
main:
.LFB10:
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
	movq	$0, _TIG_IZ_0xkN_envp(%rip)
	nop
.L44:
	movq	$0, _TIG_IZ_0xkN_argv(%rip)
	nop
.L45:
	movl	$0, _TIG_IZ_0xkN_argc(%rip)
	nop
	nop
.L46:
.L47:
#APP
# 88 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-0xkN--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_0xkN_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_0xkN_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_0xkN_envp(%rip)
	nop
	movq	$0, -24(%rbp)
.L71:
	cmpq	$19, -24(%rbp)
	ja	.L74
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L50(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L50(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L50:
	.long	.L60-.L50
	.long	.L59-.L50
	.long	.L58-.L50
	.long	.L74-.L50
	.long	.L57-.L50
	.long	.L74-.L50
	.long	.L56-.L50
	.long	.L74-.L50
	.long	.L55-.L50
	.long	.L74-.L50
	.long	.L74-.L50
	.long	.L54-.L50
	.long	.L53-.L50
	.long	.L74-.L50
	.long	.L52-.L50
	.long	.L74-.L50
	.long	.L74-.L50
	.long	.L74-.L50
	.long	.L51-.L50
	.long	.L49-.L50
	.text
.L51:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %edx
	leaq	-16(%rbp), %rax
	movd	%edx, %xmm0
	movq	%rax, %rdi
	call	withdraw
	movq	$4, -24(%rbp)
	jmp	.L61
.L57:
	movl	-32(%rbp), %eax
	cmpl	$4, %eax
	je	.L62
	movq	$12, -24(%rbp)
	jmp	.L61
.L62:
	movq	$19, -24(%rbp)
	jmp	.L61
.L52:
	movl	-32(%rbp), %eax
	cmpl	$4, %eax
	je	.L64
	cmpl	$4, %eax
	jg	.L65
	cmpl	$3, %eax
	je	.L66
	cmpl	$3, %eax
	jg	.L65
	cmpl	$1, %eax
	je	.L67
	cmpl	$2, %eax
	je	.L68
	jmp	.L65
.L64:
	movq	$11, -24(%rbp)
	jmp	.L69
.L66:
	movq	$1, -24(%rbp)
	jmp	.L69
.L68:
	movq	$18, -24(%rbp)
	jmp	.L69
.L67:
	movq	$8, -24(%rbp)
	jmp	.L69
.L65:
	movq	$6, -24(%rbp)
	nop
.L69:
	jmp	.L61
.L53:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$14, -24(%rbp)
	jmp	.L61
.L55:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %edx
	leaq	-16(%rbp), %rax
	movd	%edx, %xmm0
	movq	%rax, %rdi
	call	deposit
	movq	$4, -24(%rbp)
	jmp	.L61
.L59:
	leaq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	balanceInquiry
	movq	$4, -24(%rbp)
	jmp	.L61
.L54:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -24(%rbp)
	jmp	.L61
.L49:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L72
	jmp	.L73
.L56:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -24(%rbp)
	jmp	.L61
.L60:
	movq	$2, -24(%rbp)
	jmp	.L61
.L58:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %edx
	leaq	-16(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	createAccount
	movq	$12, -24(%rbp)
	jmp	.L61
.L74:
	nop
.L61:
	jmp	.L71
.L73:
	call	__stack_chk_fail@PLT
.L72:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
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
