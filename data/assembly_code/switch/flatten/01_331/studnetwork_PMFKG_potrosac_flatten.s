	.file	"studnetwork_PMFKG_potrosac_flatten.c"
	.text
	.globl	_TIG_IZ_3Hhm_argv
	.bss
	.align 8
	.type	_TIG_IZ_3Hhm_argv, @object
	.size	_TIG_IZ_3Hhm_argv, 8
_TIG_IZ_3Hhm_argv:
	.zero	8
	.globl	_TIG_IZ_3Hhm_argc
	.align 4
	.type	_TIG_IZ_3Hhm_argc, @object
	.size	_TIG_IZ_3Hhm_argc, 4
_TIG_IZ_3Hhm_argc:
	.zero	4
	.globl	_TIG_IZ_3Hhm_envp
	.align 8
	.type	_TIG_IZ_3Hhm_envp, @object
	.size	_TIG_IZ_3Hhm_envp, 8
_TIG_IZ_3Hhm_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Potrosac dete: %d\t%d\n"
.LC1:
	.string	"Potrosac roditelj: %d\t%d\n"
.LC2:
	.string	"./proizvodjac.c"
.LC3:
	.string	"Greska pri forkovanju!!\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_3Hhm_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_3Hhm_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_3Hhm_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 150 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-3Hhm--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_3Hhm_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_3Hhm_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_3Hhm_envp(%rip)
	nop
	movq	$1, -40(%rbp)
.L18:
	cmpq	$9, -40(%rbp)
	ja	.L20
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
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L20-.L8
	.long	.L20-.L8
	.long	.L20-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L20-.L8
	.long	.L7-.L8
	.text
.L12:
	movq	$5, -40(%rbp)
	jmp	.L14
.L7:
	leaq	-32(%rbp), %rsi
	movl	-56(%rbp), %eax
	movl	$0, %r8d
	movl	$1, %ecx
	movl	$16, %edx
	movl	%eax, %edi
	call	msgrcv@PLT
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -40(%rbp)
	jmp	.L14
.L10:
	leaq	-32(%rbp), %rsi
	movl	-56(%rbp), %eax
	movl	$0, %r8d
	movl	$1, %ecx
	movl	$16, %edx
	movl	%eax, %edi
	call	msgrcv@PLT
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -40(%rbp)
	jmp	.L14
.L11:
	movl	$0, -48(%rbp)
	movl	$20, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	ftok@PLT
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movl	$950, %esi
	movl	%eax, %edi
	call	msgget@PLT
	movl	%eax, -56(%rbp)
	call	fork@PLT
	movl	%eax, -52(%rbp)
	movq	$0, -40(%rbp)
	jmp	.L14
.L13:
	cmpl	$-1, -52(%rbp)
	je	.L15
	cmpl	$0, -52(%rbp)
	jne	.L16
	movq	$9, -40(%rbp)
	jmp	.L17
.L15:
	movq	$7, -40(%rbp)
	jmp	.L17
.L16:
	movq	$6, -40(%rbp)
	nop
.L17:
	jmp	.L14
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$0, %edi
	call	exit@PLT
.L20:
	nop
.L14:
	jmp	.L18
	.cfi_endproc
.LFE3:
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
