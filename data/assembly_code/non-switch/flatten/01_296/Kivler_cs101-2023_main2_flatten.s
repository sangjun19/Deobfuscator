	.file	"Kivler_cs101-2023_main2_flatten.c"
	.text
	.globl	_TIG_IZ_vGeg_argc
	.bss
	.align 4
	.type	_TIG_IZ_vGeg_argc, @object
	.size	_TIG_IZ_vGeg_argc, 4
_TIG_IZ_vGeg_argc:
	.zero	4
	.globl	_TIG_IZ_vGeg_envp
	.align 8
	.type	_TIG_IZ_vGeg_envp, @object
	.size	_TIG_IZ_vGeg_envp, 8
_TIG_IZ_vGeg_envp:
	.zero	8
	.globl	_TIG_IZ_vGeg_argv
	.align 8
	.type	_TIG_IZ_vGeg_argv, @object
	.size	_TIG_IZ_vGeg_argv, 8
_TIG_IZ_vGeg_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"w"
.LC1:
	.string	"main2.txt"
.LC2:
	.string	"r"
.LC3:
	.string	"main2.c"
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
	subq	$1088, %rsp
	movl	%edi, -1060(%rbp)
	movq	%rsi, -1072(%rbp)
	movq	%rdx, -1080(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_vGeg_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_vGeg_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_vGeg_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-vGeg--0
# 0 "" 2
#NO_APP
	movl	-1060(%rbp), %eax
	movl	%eax, _TIG_IZ_vGeg_argc(%rip)
	movq	-1072(%rbp), %rax
	movq	%rax, _TIG_IZ_vGeg_argv(%rip)
	movq	-1080(%rbp), %rax
	movq	%rax, _TIG_IZ_vGeg_envp(%rip)
	nop
	movq	$1, -1024(%rbp)
.L19:
	cmpq	$10, -1024(%rbp)
	ja	.L22
	movq	-1024(%rbp), %rax
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
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L22-.L8
	.long	.L12-.L8
	.long	.L22-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L22-.L8
	.long	.L9-.L8
	.long	.L22-.L8
	.long	.L7-.L8
	.text
.L9:
	movq	-1048(%rbp), %rdx
	leaq	-1008(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	-1048(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$10, -1024(%rbp)
	jmp	.L15
.L13:
	movq	$5, -1024(%rbp)
	jmp	.L15
.L12:
	movl	-1052(%rbp), %eax
	cltq
	leaq	-1008(%rbp), %rdx
	addq	%rdx, %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1016(%rbp)
	movq	-1016(%rbp), %rax
	movl	%eax, %edx
	movl	-1052(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -1052(%rbp)
	movq	$6, -1024(%rbp)
	jmp	.L15
.L10:
	movl	$1000, %eax
	subl	-1052(%rbp), %eax
	movl	-1052(%rbp), %edx
	movslq	%edx, %rdx
	leaq	-1008(%rbp), %rcx
	addq	%rdx, %rcx
	movq	-1040(%rbp), %rdx
	movl	%eax, %esi
	movq	%rcx, %rdi
	call	fgets@PLT
	movq	%rax, -1032(%rbp)
	movq	$0, -1024(%rbp)
	jmp	.L15
.L11:
	movl	$0, -1052(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1048(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1040(%rbp)
	movq	$6, -1024(%rbp)
	jmp	.L15
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L20
	jmp	.L21
.L14:
	cmpq	$0, -1032(%rbp)
	je	.L17
	movq	$3, -1024(%rbp)
	jmp	.L15
.L17:
	movq	$8, -1024(%rbp)
	jmp	.L15
.L22:
	nop
.L15:
	jmp	.L19
.L21:
	call	__stack_chk_fail@PLT
.L20:
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
