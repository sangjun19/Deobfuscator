	.file	"Surya4785_C-programming_insertionsort_flatten.c"
	.text
	.globl	_TIG_IZ_trxE_envp
	.bss
	.align 8
	.type	_TIG_IZ_trxE_envp, @object
	.size	_TIG_IZ_trxE_envp, 8
_TIG_IZ_trxE_envp:
	.zero	8
	.globl	_TIG_IZ_trxE_argv
	.align 8
	.type	_TIG_IZ_trxE_argv, @object
	.size	_TIG_IZ_trxE_argv, 8
_TIG_IZ_trxE_argv:
	.zero	8
	.globl	_TIG_IZ_trxE_argc
	.align 4
	.type	_TIG_IZ_trxE_argc, @object
	.size	_TIG_IZ_trxE_argc, 4
_TIG_IZ_trxE_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"Enter number of elements:"
.LC2:
	.string	"Enter %d integers:\n"
.LC3:
	.string	"%d\n"
	.align 8
.LC4:
	.string	"Sorted list in ascending order:"
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
	subq	$336, %rsp
	movl	%edi, -308(%rbp)
	movq	%rsi, -320(%rbp)
	movq	%rdx, -328(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_trxE_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_trxE_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_trxE_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 129 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-trxE--0
# 0 "" 2
#NO_APP
	movl	-308(%rbp), %eax
	movl	%eax, _TIG_IZ_trxE_argc(%rip)
	movq	-320(%rbp), %rax
	movq	%rax, _TIG_IZ_trxE_argv(%rip)
	movq	-328(%rbp), %rax
	movq	%rax, _TIG_IZ_trxE_envp(%rip)
	nop
	movq	$6, -280(%rbp)
.L35:
	cmpq	$27, -280(%rbp)
	ja	.L38
	movq	-280(%rbp), %rax
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
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L38-.L8
	.long	.L17-.L8
	.long	.L38-.L8
	.long	.L16-.L8
	.long	.L38-.L8
	.long	.L15-.L8
	.long	.L38-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L12-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L11-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L7-.L8
	.text
.L18:
	leaq	-272(%rbp), %rdx
	movl	-292(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -292(%rbp)
	movq	$16, -280(%rbp)
	jmp	.L23
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L36
	jmp	.L37
.L16:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-296(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-296(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -292(%rbp)
	movq	$16, -280(%rbp)
	jmp	.L23
.L21:
	movl	$1, -292(%rbp)
	movq	$10, -280(%rbp)
	jmp	.L23
.L10:
	cmpl	$0, -288(%rbp)
	jle	.L25
	movq	$24, -280(%rbp)
	jmp	.L23
.L25:
	movq	$27, -280(%rbp)
	jmp	.L23
.L19:
	movl	-292(%rbp), %eax
	cltq
	movl	-272(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -292(%rbp)
	movq	$19, -280(%rbp)
	jmp	.L23
.L12:
	movl	-296(%rbp), %eax
	cmpl	%eax, -292(%rbp)
	jge	.L27
	movq	$4, -280(%rbp)
	jmp	.L23
.L27:
	movq	$1, -280(%rbp)
	jmp	.L23
.L9:
	movl	-288(%rbp), %eax
	subl	$1, %eax
	cltq
	movl	-272(%rbp,%rax,4), %edx
	movl	-288(%rbp), %eax
	cltq
	movl	-272(%rbp,%rax,4), %eax
	cmpl	%eax, %edx
	jle	.L29
	movq	$2, -280(%rbp)
	jmp	.L23
.L29:
	movq	$27, -280(%rbp)
	jmp	.L23
.L13:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -292(%rbp)
	movq	$19, -280(%rbp)
	jmp	.L23
.L11:
	movl	-296(%rbp), %eax
	cmpl	%eax, -292(%rbp)
	jge	.L31
	movq	$3, -280(%rbp)
	jmp	.L23
.L31:
	movq	$12, -280(%rbp)
	jmp	.L23
.L17:
	movq	$8, -280(%rbp)
	jmp	.L23
.L7:
	addl	$1, -292(%rbp)
	movq	$10, -280(%rbp)
	jmp	.L23
.L15:
	movl	-296(%rbp), %eax
	cmpl	%eax, -292(%rbp)
	jge	.L33
	movq	$0, -280(%rbp)
	jmp	.L23
.L33:
	movq	$13, -280(%rbp)
	jmp	.L23
.L22:
	movl	-292(%rbp), %eax
	movl	%eax, -288(%rbp)
	movq	$23, -280(%rbp)
	jmp	.L23
.L20:
	movl	-288(%rbp), %eax
	cltq
	movl	-272(%rbp,%rax,4), %eax
	movl	%eax, -284(%rbp)
	movl	-288(%rbp), %eax
	subl	$1, %eax
	cltq
	movl	-272(%rbp,%rax,4), %edx
	movl	-288(%rbp), %eax
	cltq
	movl	%edx, -272(%rbp,%rax,4)
	movl	-288(%rbp), %eax
	subl	$1, %eax
	cltq
	movl	-284(%rbp), %edx
	movl	%edx, -272(%rbp,%rax,4)
	subl	$1, -288(%rbp)
	movq	$23, -280(%rbp)
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
